"""Збереження у pickle та JSON з перевіркою структури після читання."""

import json
import pickle
from array import array
from contextlib import contextmanager
from pathlib import Path

from findex.models import ArrayPostings, DocMeta, Index, PlainPosting, Posting, pairs
from findex.timing import timed

VERSION = 2


def file_format(path: Path, format: str | None) -> str:
    selected = format or ("json" if path.suffix.lower() == ".json" else "pickle")
    if selected not in ("pickle", "json"):
        raise ValueError(f"Невідомий формат: {selected}")
    return selected


def validate(index: Index) -> None:
    """Відсіяти несумісний або пошкоджений індекс; це не захист від pickle."""
    if not isinstance(index, Index) or index.storage not in ("plain", "slots", "array"):
        raise ValueError("Непідтримувана структура індексу")
    ids = set(index.doc_meta)
    if ids != set(index.doc_lengths) or ids != set(range(len(ids))):
        raise ValueError("Ідентифікатори документів мають бути послідовними від нуля")
    for doc_id, length in index.doc_lengths.items():
        if type(doc_id) is not int or type(length) is not int or length < 0:
            raise ValueError("Некоректна довжина документа")
        meta = index.doc_meta[doc_id]
        if not isinstance(meta, DocMeta) or not all(
            isinstance(value, str) for value in (meta.path, meta.title)
        ):
            raise ValueError("Некоректні метадані")
    totals = dict.fromkeys(ids, 0)
    for term, postings in index.postings.items():
        if not isinstance(term, str) or not term or not len(postings):
            raise ValueError("Некоректний термін або порожній список постінгів")
        previous = -1
        for doc_id, tf in pairs(postings):
            if (
                type(doc_id) is not int
                or type(tf) is not int
                or doc_id not in ids
                or doc_id <= previous
                or tf <= 0
            ):
                raise ValueError("Постінги мають бути унікальними та відсортованими")
            totals[doc_id] += tf
            previous = doc_id
    if totals != index.doc_lengths:
        raise ValueError("Частоти не збігаються з довжинами документів")
    if any(
        doc not in ids or not isinstance(text, str) for doc, text in index.texts.items()
    ):
        raise ValueError("Некоректні тексти для снипетів")
    if index.has_positions:
        if set(index._positions) != set(index):
            raise ValueError("Позиції не відповідають словнику")
        for term, postings in index.items():
            if set(index._positions[term]) != {p.doc_id for p in postings}:
                raise ValueError("Позиції не відповідають документам терміна")
            for posting in postings:
                positions = index.positions(term, posting.doc_id)
                if (
                    len(positions) != posting.tf
                    or any(
                        type(p) is not int
                        or not 0 <= p < index.doc_length(posting.doc_id)
                        for p in positions
                    )
                    or any(a >= b for a, b in zip(positions, positions[1:]))
                ):
                    raise ValueError("Некоректні позиції токенів")


def save(index: Index, path: Path, format: str | None = None) -> None:
    selected = file_format(path, format)
    if selected == "pickle":
        with path.open("wb") as target:
            pickle.dump({"version": VERSION, "index": index}, target, protocol=5)
    else:
        data = {
            "version": VERSION,
            "storage": index.storage,
            "documents": [
                [doc_id, index.doc_lengths[doc_id], meta.path, meta.title]
                for doc_id, meta in index.doc_meta.items()
            ],
            "postings": {
                term: list(pairs(items)) for term, items in index.postings.items()
            },
            "texts": dict(index.texts),
            "positions": index._positions,
        }
        with path.open("w", encoding="utf-8") as target:
            json.dump(data, target, ensure_ascii=False, separators=(",", ":"))


def from_json(data: dict) -> Index:
    storage = data["storage"]
    if storage not in ("plain", "slots", "array"):
        raise ValueError("Невідомий спосіб зберігання")
    lengths = {}
    metadata = {}
    for doc_id, length, path, title in data["documents"]:
        if type(doc_id) is not int or doc_id in metadata:
            raise ValueError("Некоректний або повторний ідентифікатор")
        lengths[doc_id] = length
        metadata[doc_id] = DocMeta(path, title)
    postings = {}
    record = PlainPosting if storage == "plain" else Posting
    for term, items in data["postings"].items():
        # Перевірка до array, щоб bool не перетворився непомітно на int.
        if any(type(doc) is not int or type(tf) is not int for doc, tf in items):
            raise ValueError("Постінг має містити два цілі числа")
        if storage == "array":
            postings[term] = ArrayPostings(
                array("I", (doc for doc, _ in items)),
                array("I", (tf for _, tf in items)),
            )
        else:
            postings[term] = [record(doc, tf) for doc, tf in items]
    positions = data.get("positions")
    if positions is not None:
        positions = {
            term: {int(doc): tuple(values) for doc, values in docs.items()}
            for term, docs in positions.items()
        }
    texts = {int(doc): text for doc, text in data.get("texts", {}).items()}
    return Index(postings, lengths, metadata, storage, positions, texts)


@timed
def load(path: Path, format: str | None = None) -> Index:
    """Pickle дозволено читати лише з довіреного власного файла."""
    selected = file_format(path, format)
    try:
        if selected == "pickle":
            with path.open("rb") as source:
                # pickle.load може виконати довільний код ще до validate().
                # Ніколи не завантажувати тут чужі або отримані з мережі файли.
                data = pickle.load(source)
        else:
            with path.open(encoding="utf-8") as source:
                data = json.load(source)
        if not isinstance(data, dict) or data.get("version") not in (1, VERSION):
            raise ValueError("Непідтримувана версія індексу")
        index = data["index"] if selected == "pickle" else from_json(data)
        validate(index)
        return index
    except (
        KeyError,
        TypeError,
        AttributeError,
        EOFError,
        pickle.UnpicklingError,
        OverflowError,
        ImportError,
    ) as error:
        raise ValueError("Пошкоджений або несумісний файл індексу") from error


@contextmanager
def open_index(path: Path, format: str | None = None):
    """Файли закриває load; finally звільняє пам'ять і кеш навіть при винятку."""
    index = load(path, format)
    try:
        yield index
    finally:
        index.close()
