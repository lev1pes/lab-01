"""Збереження у pickle та JSON з перевіркою структури після читання."""

import json
import pickle
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast

from findex.models import (
    ArrayPostings,
    DocMeta,
    Index,
    PlainPosting,
    Positions,
    Posting,
    PostingList,
    pairs,
)
from findex.timing import timed

VERSION = 2
type FileFormat = Literal["pickle", "json"]


def file_format(path: Path, format: FileFormat | None) -> FileFormat:
    selected = format or ("json" if path.suffix.lower() == ".json" else "pickle")
    if selected not in ("pickle", "json"):
        raise ValueError(f"Невідомий формат: {selected}")
    return selected


def validate(index: object) -> None:
    """Відсіяти несумісний або пошкоджений індекс; це не захист від pickle."""
    if not isinstance(index, Index) or index.storage not in ("plain", "slots", "array"):
        raise ValueError("Непідтримувана структура індексу")
    # Поля довіреного pickle теж можуть бути пошкодженими: перевіряємо як object.
    ids = set(index.doc_meta)
    if ids != set(index.doc_lengths) or ids != set(range(len(ids))):
        raise ValueError("Ідентифікатори документів мають бути послідовними від нуля")
    for doc_id, length in index.doc_lengths.items():
        if type(doc_id) is not int or type(length) is not int or length < 0:
            raise ValueError("Некоректна довжина документа")
        meta = index.doc_meta[doc_id]
        if not isinstance(cast(object, meta), DocMeta) or not all(
            isinstance(cast(object, value), str) for value in (meta.path, meta.title)
        ):
            raise ValueError("Некоректні метадані")
    totals = dict.fromkeys(ids, 0)
    for term, postings in index.postings.items():
        if not isinstance(cast(object, term), str) or not term or not len(postings):
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
        doc not in ids or not isinstance(cast(object, text), str)
        for doc, text in index.texts.items()
    ):
        raise ValueError("Некоректні тексти для снипетів")
    positions_by_term = index.__getstate__()["positions"]
    if positions_by_term is not None:
        if set(positions_by_term) != set(index):
            raise ValueError("Позиції не відповідають словнику")
        for term, postings in index.items():
            if set(positions_by_term[term]) != {p.doc_id for p in postings}:
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


def save(index: Index, path: Path, format: FileFormat | None = None) -> None:
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
            "positions": index.__getstate__()["positions"],
        }
        with path.open("w", encoding="utf-8") as target:
            json.dump(data, target, ensure_ascii=False, separators=(",", ":"))


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Очікувався JSON-об'єкт")
    # Перевіряємо контейнер і ключі; значення перевіряються окремо нижче.
    result = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in result):
        raise ValueError("Ключ JSON має бути рядком")
    return cast(dict[str, object], result)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Очікувався JSON-масив")
    # Елементи ще не перевірені, тому їхній тип залишається object.
    return cast(list[object], value)


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Очікувалося ціле число, не bool")
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Очікувався рядок")
    return value


def from_json(data: dict[str, object]) -> Index:
    storage = data["storage"]
    if storage not in ("plain", "slots", "array"):
        raise ValueError("Невідомий спосіб зберігання")
    lengths: dict[int, int] = {}
    metadata: dict[int, DocMeta] = {}
    for row in _list(data["documents"]):
        raw_id, raw_length, raw_path, raw_title = _list(row)
        doc_id = _integer(raw_id)
        if doc_id in metadata:
            raise ValueError("Повторний ідентифікатор")
        lengths[doc_id] = _integer(raw_length)
        metadata[doc_id] = DocMeta(_string(raw_path), _string(raw_title))
    postings: dict[str, PostingList] = {}
    record = PlainPosting if storage == "plain" else Posting
    for term, raw_items in _mapping(data["postings"]).items():
        items: list[tuple[int, int]] = []
        for row in _list(raw_items):
            doc, tf = _list(row)
            items.append((_integer(doc), _integer(tf)))
        if storage == "array":
            postings[term] = ArrayPostings(
                array("I", (doc for doc, _ in items)),
                array("I", (tf for _, tf in items)),
            )
        else:
            postings[term] = [record(doc, tf) for doc, tf in items]
    raw_positions = data.get("positions")
    positions: Positions | None = None
    if raw_positions is not None:
        positions = {
            term: {
                int(doc): tuple(_integer(p) for p in _list(values))
                for doc, values in _mapping(docs).items()
            }
            for term, docs in _mapping(raw_positions).items()
        }
    texts = {
        int(doc): _string(text) for doc, text in _mapping(data.get("texts", {})).items()
    }
    return Index(postings, lengths, metadata, storage, positions, texts)


@timed
def load(path: Path, format: FileFormat | None = None) -> Index:
    """Pickle дозволено читати лише з довіреного власного файла."""
    selected = file_format(path, format)
    try:
        if selected == "pickle":
            with path.open("rb") as source:
                # pickle.load може виконати довільний код ще до validate().
                # Ніколи не завантажувати тут чужі або отримані з мережі файли.
                raw: object = pickle.load(source)
        else:
            with path.open(encoding="utf-8") as source:
                raw = json.load(source)
        data = _mapping(raw)
        if data.get("version") not in (1, VERSION):
            raise ValueError("Непідтримувана версія індексу")
        index = data["index"] if selected == "pickle" else from_json(data)
        validate(index)
        if not isinstance(index, Index):
            raise ValueError("Очікувався індекс")
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
def open_index(
    path: Path, format: FileFormat | None = None
) -> Generator[Index, None, None]:
    """Файли закриває load; finally звільняє пам'ять і кеш навіть при винятку."""
    index = load(path, format)
    try:
        yield index
    finally:
        index.close()
