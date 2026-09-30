"""MiniLM через ONNX, фрагменти, cosine і Reciprocal Rank Fusion."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from findex.models import Index
from findex.scoring import BM25, Scorer
from findex.search import SearchResult, search
from findex.tokenize import tokenize
from findex.vector import rank_numpy, top_indices

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
type Mode = Literal["keyword", "semantic", "hybrid"]


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


class EmbeddingBackend(Protocol):
    def embed(
        self, texts: list[str], *, batch_size: int
    ) -> Iterable[NDArray[np.float32]]: ...


class MiniLM:
    """Модель створюється один раз; threads=1 не множить потоки HTTP-воркерів."""

    def __init__(
        self, model: str = MODEL, cache_dir: Path | None = None, threads: int = 1
    ) -> None:
        if model != MODEL:
            raise ValueError(f"Підтримувана модель: {MODEL}")
        try:
            module = importlib.import_module("fastembed")
        except ImportError:
            raise ValueError(
                "Встановіть семантичний модуль: uv sync --extra semantic"
            ) from None
        factory = cast(Callable[..., EmbeddingBackend], module.TextEmbedding)
        self.backend = factory(
            model_name=model,
            cache_dir=str(cache_dir) if cache_dir else None,
            threads=threads,
            enable_cpu_mem_arena=False,
        )
        self.lock = Lock()

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        with self.lock:
            return np.asarray(
                list(self.backend.embed(texts, batch_size=16)), dtype=np.float32
            )


def chunks(text: str, size: int = 160, overlap: int = 32) -> Iterator[str]:
    """Слова з overlap; MiniLM додатково обрізає до 256 wordpiece-токенів."""
    if not 0 <= overlap < size:
        raise ValueError("Потрібно 0 <= overlap < size")
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield " ".join(words[start : start + size])
        if start + size >= len(words):
            break


def fingerprint(index: Index) -> str:
    digest = hashlib.sha256()
    for doc in sorted(index.doc_meta):
        digest.update(
            json.dumps(
                [doc, index.doc_meta[doc].path, index.texts.get(doc, "")],
                ensure_ascii=False,
            ).encode()
        )
    return digest.hexdigest()


def normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if not np.isfinite(matrix).all() or (norms <= 0).any():
        raise ValueError("Ембеддинги мають бути скінченними ненульовими векторами")
    return np.asarray(matrix / norms, dtype=np.float32)


@dataclass
class SemanticIndex:
    vectors: NDArray[np.float32]
    doc_ids: NDArray[np.int32]
    passages: list[str]
    corpus_hash: str
    model: str = MODEL

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors, allow_pickle=False)
        np.save(directory / "doc_ids.npy", self.doc_ids, allow_pickle=False)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "model": self.model,
                    "corpus_hash": self.corpus_hash,
                    "passages": self.passages,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path, index: Index) -> SemanticIndex:
        metadata = cast(
            dict[str, object],
            json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
        )
        vectors = cast(
            NDArray[np.float32], np.load(directory / "vectors.npy", allow_pickle=False)
        )
        ids = cast(
            NDArray[np.int32], np.load(directory / "doc_ids.npy", allow_pickle=False)
        )
        passages = metadata.get("passages")
        if (
            metadata.get("version") != 1
            or metadata.get("model") != MODEL
            or metadata.get("corpus_hash") != fingerprint(index)
        ):
            raise ValueError("Ембеддинги не відповідають моделі або корпусу")
        if (
            vectors.dtype != np.float32
            or vectors.ndim != 2
            or vectors.shape[1] != 384
            or ids.dtype != np.int32
            or ids.shape != (len(vectors),)
            or not isinstance(passages, list)
            or len(cast(list[object], passages)) != len(ids)
            or any(not isinstance(p, str) for p in cast(list[object], passages))
            or (ids < 0).any()
            or (ids >= index.num_docs).any()
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        ):
            raise ValueError("Пошкоджені масиви ембеддингів")
        return cls(vectors, ids, cast(list[str], passages), fingerprint(index))

    def rank(self, index: Index, query: str, encoder: Encoder) -> list[SearchResult]:
        if not len(self.vectors):
            return []
        q = normalize(encoder.encode([query]))
        if q.shape != (1, self.vectors.shape[1]):
            raise ValueError("Розмірність вектора запиту не відповідає індексу")
        similarities = self.vectors @ q[0]
        scores = np.full(index.num_docs, -np.inf, dtype=np.float64)
        np.maximum.at(scores, self.doc_ids, similarities)
        ids = np.flatnonzero(np.isfinite(scores)).astype(np.int32)
        selected = top_indices(scores[ids], ids, len(ids))
        best: dict[int, int] = {}
        for i in np.flatnonzero(similarities == scores[self.doc_ids]):
            best.setdefault(int(self.doc_ids[i]), int(i))
        return [
            SearchResult(
                int(doc),
                float(scores[doc]),
                index.doc_meta[int(doc)].title,
                self.passages[best[int(doc)]][:240],
            )
            for doc in ids[selected]
        ]


def embed(
    index: Index,
    encoder: Encoder,
    progress: Callable[[int], None] | None = None,
    checkpoint_dir: Path | None = None,
) -> SemanticIndex:
    def encode_batch(batch: list[str]) -> NDArray[np.float32]:
        key = hashlib.sha256(
            json.dumps([MODEL, batch], ensure_ascii=False).encode()
        ).hexdigest()
        path = checkpoint_dir / (key + ".npy") if checkpoint_dir else None
        if path is not None and path.exists():
            cached = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            if cached.shape != (len(batch), 384):
                raise ValueError("Пошкоджений checkpoint ембеддингів")
            return normalize(cached)
        result = normalize(encoder.encode(batch))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as target:
                np.save(target, result, allow_pickle=False)
            temporary.replace(path)
        return result

    passages: list[str] = []
    ids: list[int] = []
    vectors: list[NDArray[np.float32]] = []
    batch: list[str] = []
    for doc in sorted(index.doc_meta):
        for passage in chunks(index.texts.get(doc, "")):
            passages.append(passage)
            ids.append(doc)
            batch.append(passage)
            if len(batch) == 16:
                vectors.append(encode_batch(batch))
                if progress:
                    progress(len(batch))
                batch = []
    if batch:
        vectors.append(encode_batch(batch))
        if progress:
            progress(len(batch))
    matrix = (
        np.concatenate(vectors) if vectors else np.empty((0, 384), dtype=np.float32)
    )
    return SemanticIndex(
        matrix, np.asarray(ids, dtype=np.int32), passages, fingerprint(index)
    )


def rrf(rankings: list[list[SearchResult]], constant: int = 60) -> list[SearchResult]:
    if constant <= 0:
        raise ValueError("RRF constant має бути додатним")
    scores: dict[int, float] = {}
    originals: dict[int, SearchResult] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for position, result in enumerate(ranking, 1):
            if result.doc_id in seen:
                continue
            seen.add(result.doc_id)
            scores[result.doc_id] = scores.get(result.doc_id, 0) + 1 / (
                constant + position
            )
            if result.doc_id not in originals or not originals[result.doc_id].snippet:
                originals[result.doc_id] = result
    return [
        SearchResult(doc, scores[doc], originals[doc].title, originals[doc].snippet)
        for doc in sorted(scores, key=lambda d: (-scores[d], d))
    ]


def retrieve(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    *,
    mode: Mode = "keyword",
    semantic: SemanticIndex | None = None,
    encoder: Encoder | None = None,
) -> list[SearchResult]:
    if not query.strip() or k < 0:
        raise ValueError("Потрібен непорожній запит і k >= 0")
    if mode == "keyword":
        return search(index, query, scorer, k)
    if mode not in ("semantic", "hybrid"):
        raise ValueError("Невідомий режим пошуку")
    if semantic is None or encoder is None:
        raise ValueError("Семантичний індекс не завантажено; виконайте findex embed")
    ranking = semantic.rank(index, query, encoder)
    if mode == "hybrid":
        # Вільне питання: OR-кандидати, не неявний AND усіх слів речення.
        lexical = " OR ".join(sorted(set(tokenize(query))))
        keyword = (
            [
                SearchResult(doc, score, index.doc_meta[doc].title)
                for doc, score in rank_numpy(index, lexical, BM25(), index.num_docs)
            ]
            if lexical
            else []
        )
        ranking = rrf([keyword, ranking])
    return ranking[:k]
