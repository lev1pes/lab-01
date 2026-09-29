"""Шляхи → незалежні часткові індекси → детерміноване злиття."""

import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Literal

from findex.corpus import Document
from findex.index import build_state
from findex.models import (
    ArrayPostings,
    Index,
    IndexState,
    NumpyPostings,
    Posting,
    Storage,
    pairs,
)

type ExecutorName = Literal["serial", "threads", "processes"]


class WorkerError(RuntimeError):
    """Причина зберігає оригінальний traceback, зокрема з іншого процесу."""


@dataclass(frozen=True, slots=True)
class PartialIndex:
    start_id: int
    state: IndexState


@dataclass(slots=True)
class BuildMetrics:
    merge_seconds: float = 0.0
    chunks: int = 0


def document_paths(root: Path, limit: int | None = None) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Каталог корпусу не знайдено: {root}")
    if limit is not None and limit < 0:
        raise ValueError("Ліміт має бути невід'ємним")
    # Той самий порядок rglob, що у завантажувачі лабораторної 4.
    paths = (p for p in root.rglob("*.txt") if p.is_file())
    return list(islice(paths, limit)) if limit is not None else list(paths)


def build_partial(
    doc_paths: Sequence[Path],
    root: Path,
    start_id: int = 0,
    storage: Storage = "numpy",
    positions: bool = False,
) -> PartialIndex:
    """Функція рівня модуля; читання й усі змінні дані належать воркеру."""
    if start_id < 0:
        raise ValueError("Початковий ID має бути невід'ємним")

    def documents() -> Iterator[Document]:
        for path in doc_paths:
            # Помилка читання не маскується пропуском документа та зміною ID.
            yield Document(
                path.relative_to(root).as_posix(),
                path,
                path.read_text(encoding="utf-8-sig"),
            )

    state = build_state(documents(), storage, positions=positions, start_id=start_id)
    return PartialIndex(start_id, state)


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Діапазони не перетинаються: конкатенація в порядку ID вже відсортована."""
    ordered = sorted(partials, key=lambda part: part.start_id)
    if not ordered:
        return Index({}, {}, {})
    first = ordered[0].state
    state = IndexState(
        postings={},
        doc_lengths={},
        doc_meta={},
        storage=first["storage"],
        positions={} if first["positions"] is not None else None,
        texts={},
    )
    for part in ordered:
        incoming = part.state
        if incoming["storage"] != state["storage"] or (
            (incoming["positions"] is None) != (state["positions"] is None)
        ):
            raise ValueError("Часткові індекси мають різні налаштування")
        expected = list(
            range(
                len(state["doc_meta"]),
                len(state["doc_meta"]) + len(incoming["doc_meta"]),
            )
        )
        if list(incoming["doc_meta"]) != expected:
            raise ValueError("Діапазони ID перетинаються або мають пропуски")
        for term, postings in incoming["postings"].items():
            if term not in state["postings"]:
                # Окремі контейнери: merge не змінює вхідні partials.
                state["postings"][term] = (
                    ArrayPostings(postings.doc_ids[:], postings.tfs[:])
                    if isinstance(postings, ArrayPostings)
                    else [Posting(d, t) for d, t in pairs(postings)]
                )
            else:
                target = state["postings"][term]
                if isinstance(target, ArrayPostings) and isinstance(
                    postings, ArrayPostings
                ):
                    target.doc_ids.extend(postings.doc_ids)
                    target.tfs.extend(postings.tfs)
                elif isinstance(target, list) and isinstance(
                    postings, (list, NumpyPostings)
                ):
                    target.extend(Posting(d, t) for d, t in pairs(postings))
                else:
                    raise ValueError("Несумісні постінги")
        for term, docs in (incoming["positions"] or {}).items():
            offsets = state["positions"]
            if offsets is not None:
                offsets.setdefault(term, {}).update(docs)
        state["doc_lengths"].update(incoming["doc_lengths"])
        state["doc_meta"].update(incoming["doc_meta"])
        state["texts"].update(incoming["texts"])
    return Index(**state)


def build_parallel(
    paths: Sequence[Path],
    root: Path,
    *,
    workers: int = 1,
    executor: ExecutorName = "serial",
    storage: Storage = "numpy",
    positions: bool = False,
    metrics: BuildMetrics | None = None,
    progress: Callable[[int], None] | None = None,
) -> Index:
    if workers < 1:
        raise ValueError("Кількість воркерів має бути додатною")
    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Невідомий виконавець: {executor}")
    if os.name == "nt" and executor == "processes" and workers > 61:
        raise ValueError("ProcessPoolExecutor у Windows підтримує до 61 воркера")
    count = min(workers, len(paths)) or 1
    size = max(1, (len(paths) + count - 1) // count)
    chunks: list[tuple[int, Sequence[Path]]] = [
        (offset, paths[offset : offset + size]) for offset in range(0, len(paths), size)
    ]
    partials: list[PartialIndex] = []
    if not chunks:
        chunks = [(0, [])]
    if executor == "serial":
        for offset, chunk in chunks:
            partials.append(build_partial(chunk, root, offset, storage, positions))
            if progress:
                progress(len(chunk))
    else:
        pool = (
            ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")
            )
            if executor == "processes"
            else ThreadPoolExecutor(workers)
        )
        with pool:
            futures = {
                pool.submit(
                    build_partial, chunk, root, offset, storage, positions
                ): len(chunk)
                for offset, chunk in chunks
            }
            try:
                for future in as_completed(futures):
                    partials.append(future.result())
                    if progress:
                        progress(futures[future])
            except Exception as error:
                for future in futures:
                    future.cancel()
                raise WorkerError("Помилка воркера під час індексації") from error
    started = perf_counter()
    index = merge(partials)
    if metrics is not None:
        metrics.merge_seconds = perf_counter() - started
        metrics.chunks = len(chunks)
    return index
