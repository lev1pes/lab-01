"""Структури інвертованого індексу та три способи зберігання постінгів."""

from __future__ import annotations

import logging
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from types import MappingProxyType
from typing import Literal, NamedTuple, TypedDict, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


@dataclass(frozen=True)
class PlainPosting:
    """Та сама інформація, але зі словником атрибутів екземпляра."""

    doc_id: int
    tf: int


@dataclass(slots=True)
class ArrayPostings:
    doc_ids: array[int]
    tfs: array[int]

    def __len__(self) -> int:
        return len(self.doc_ids)


@dataclass(frozen=True, slots=True)
class NumpyPostings:
    doc_ids: NDArray[np.int32]
    tfs: NDArray[np.int32]

    def __len__(self) -> int:
        return len(self.doc_ids)


def compact(items: PostingList) -> NumpyPostings:
    if isinstance(items, NumpyPostings):
        return items
    values = list(pairs(items))
    if any(not 0 <= d <= 2**31 - 1 or not 0 < t <= 2**31 - 1 for d, t in values):
        raise ValueError("Постінг не вміщується в int32")
    ids = np.array([d for d, _ in values], dtype=np.int32)
    tfs = np.array([t for _, t in values], dtype=np.int32)
    ids.flags.writeable = tfs.flags.writeable = False
    return NumpyPostings(ids, tfs)


type Storage = Literal["plain", "slots", "array", "numpy"]
type PostingList = list[Posting | PlainPosting] | ArrayPostings | NumpyPostings
type Positions = dict[str, dict[int, tuple[int, ...]]]


class IndexState(TypedDict):
    postings: dict[str, PostingList]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    storage: Storage
    positions: Positions | None
    texts: dict[int, str]


class CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class Index(Mapping[str, tuple[Posting, ...]]):
    """Завершений індекс: доступ лише для читання, власний кеш запитів."""

    def __init__(
        self,
        postings: dict[str, PostingList],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        storage: Storage = "numpy",
        positions: Positions | None = None,
        texts: dict[int, str] | None = None,
    ) -> None:
        self._postings: dict[str, PostingList] = (
            {term: compact(items) for term, items in postings.items()}
            if storage == "numpy"
            else postings
        )
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self.storage: Storage = storage
        self._positions = positions
        self._texts = texts or {}
        self.closed = False

        # Ключ кешу — лише рядок; індекси не ділять відповіді між собою.
        @lru_cache(maxsize=256)
        def cached_ids(query: str) -> frozenset[int]:
            return self._evaluate_query(query)

        self._cached_ids = cached_ids

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("Індекс уже закрито")

    def __len__(self) -> int:
        self._check_open()
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        self._check_open()
        return iter(self._postings)

    def __contains__(self, term: object) -> bool:
        self._check_open()
        return term in self._postings

    def __getitem__(self, term: str) -> tuple[Posting, ...]:
        self._check_open()
        # Незмінний знімок не дозволяє зіпсувати індекс через публічний API.
        return tuple(Posting(doc, tf) for doc, tf in pairs(self._postings[term]))

    def __repr__(self) -> str:
        if self.closed:
            return "Index(closed=True)"
        return f"Index(terms={len(self)}, docs={self.num_docs})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return (
            dict(self.items()) == dict(other.items())
            and self.doc_lengths == other.doc_lengths
            and self.doc_meta == other.doc_meta
            and self._positions == other._positions
            and self.texts == other.texts
            and self.storage == other.storage
        )

    @property
    def num_docs(self) -> int:
        self._check_open()
        return len(self._doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    @cached_property
    def length_array(self) -> NDArray[np.int32]:
        if any(not 0 <= n <= 2**31 - 1 for n in self._doc_lengths.values()):
            raise ValueError("Довжина не вміщується в int32")
        result = np.array(
            [self._doc_lengths[i] for i in range(self.num_docs)], dtype=np.int32
        )
        result.flags.writeable = False
        return result

    @cached_property
    def numpy_postings(self) -> dict[str, NumpyPostings]:
        self._check_open()
        return {term: compact(items) for term, items in self._postings.items()}

    def df(self, term: str) -> int:
        self._check_open()
        return len(self._postings.get(term, ()))

    @property
    def postings(self) -> Index:
        """Сумісний із лабораторною 2 інтерфейс словника."""
        self._check_open()
        return self

    @property
    def doc_lengths(self) -> Mapping[int, int]:
        self._check_open()
        return MappingProxyType(self._doc_lengths)

    @property
    def doc_meta(self) -> Mapping[int, DocMeta]:
        self._check_open()
        return MappingProxyType(self._doc_meta)

    @property
    def texts(self) -> Mapping[int, str]:
        self._check_open()
        return MappingProxyType(self._texts)

    @property
    def has_positions(self) -> bool:
        self._check_open()
        return self._positions is not None

    def positions(self, term: str, doc_id: int) -> tuple[int, ...]:
        self._check_open()
        if self._positions is None:
            raise ValueError("Фразовий пошук потребує побудови з --positions")
        return self._positions.get(term, {}).get(doc_id, ())

    def _evaluate_query(self, query: str) -> frozenset[int]:
        from findex.query import parse

        return frozenset(parse(query).evaluate(self))

    def matched_ids(self, query: str) -> frozenset[int]:
        self._check_open()
        before = self._cached_ids.cache_info()
        result = self._cached_ids(query)
        info = self._cached_ids.cache_info()
        logging.getLogger("findex").info(
            "cache %s: hits=%d misses=%d query=%r",
            "hit" if info.hits > before.hits else "miss",
            info.hits,
            info.misses,
            query,
        )
        return result

    def cache_info(self) -> CacheInfo:
        return CacheInfo(*self._cached_ids.cache_info())

    def close(self) -> None:
        """Звільнити власні буфери й кеш; зовнішні знімки належать викликачеві."""
        self._cached_ids.cache_clear()
        self._postings.clear()
        self._doc_lengths.clear()
        self._doc_meta.clear()
        self._texts.clear()
        if self._positions is not None:
            self._positions.clear()
        self.__dict__.pop("avg_doc_length", None)
        self.__dict__.pop("length_array", None)
        self.__dict__.pop("numpy_postings", None)
        self.closed = True

    def __getstate__(self) -> IndexState:
        self._check_open()
        return IndexState(
            postings=self._postings,
            doc_lengths=self._doc_lengths,
            doc_meta=self._doc_meta,
            storage=self.storage,
            positions=self._positions,
            texts=self._texts,
        )

    def __setstate__(self, state: object) -> None:
        # Старий slotted Index лабораторної 2 мав стан (None, словник).
        if isinstance(state, tuple):
            state = cast(tuple[object, object], state)[1]
        # Pickle лише довірений; load додатково перевіряє відновлений індекс.
        restored = cast(IndexState, state)
        self.__init__(**restored)


def pairs(
    postings: Sequence[Posting | PlainPosting] | ArrayPostings | NumpyPostings,
) -> Iterator[tuple[int, int]]:
    """Однаковий інтерфейс для об'єктів та компактних масивів."""
    if isinstance(postings, (ArrayPostings, NumpyPostings)):
        for doc, tf in zip(postings.doc_ids, postings.tfs, strict=True):
            yield int(doc), int(tf)
    else:
        for posting in postings:
            yield posting.doc_id, posting.tf


def document_ids(
    postings: Sequence[Posting | PlainPosting] | ArrayPostings | NumpyPostings,
) -> list[int]:
    if isinstance(postings, (ArrayPostings, NumpyPostings)):
        return [int(doc) for doc in postings.doc_ids]
    return [posting.doc_id for posting in postings]
