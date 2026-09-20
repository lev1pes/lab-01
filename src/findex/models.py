"""Структури інвертованого індексу та три способи зберігання постінгів."""

import logging
from array import array
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from types import MappingProxyType


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
    doc_ids: array
    tfs: array

    def __len__(self) -> int:
        return len(self.doc_ids)


type PostingList = list[Posting] | list[PlainPosting] | ArrayPostings


class Index(Mapping[str, tuple[Posting, ...]]):
    """Завершений індекс: доступ лише для читання, власний кеш запитів."""

    def __init__(
        self,
        postings,
        doc_lengths,
        doc_meta,
        storage="slots",
        positions=None,
        texts=None,
    ):
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self.storage = storage
        self._positions = positions
        self._texts = texts or {}
        self.closed = False

        # Ключ кешу — лише рядок; індекси не ділять відповіді між собою.
        @lru_cache(maxsize=256)
        def cached_ids(query):
            return self._evaluate_query(query)

        self._cached_ids = cached_ids

    def _check_open(self):
        if self.closed:
            raise ValueError("Індекс уже закрито")

    def __len__(self):
        self._check_open()
        return len(self._postings)

    def __iter__(self):
        self._check_open()
        return iter(self._postings)

    def __contains__(self, term):
        self._check_open()
        return term in self._postings

    def __getitem__(self, term):
        self._check_open()
        # Незмінний знімок не дозволяє зіпсувати індекс через публічний API.
        return tuple(Posting(doc, tf) for doc, tf in pairs(self._postings[term]))

    def __repr__(self):
        if self.closed:
            return "Index(closed=True)"
        return f"Index(terms={len(self)}, docs={self.num_docs})"

    def __eq__(self, other):
        if not isinstance(other, Index):
            return NotImplemented
        return self.__getstate__() == other.__getstate__()

    @property
    def num_docs(self):
        self._check_open()
        return len(self._doc_lengths)

    @cached_property
    def avg_doc_length(self):
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id):
        return self.doc_lengths[doc_id]

    def df(self, term):
        self._check_open()
        return len(self._postings.get(term, ()))

    @property
    def postings(self):
        """Сумісний із лабораторною 2 інтерфейс словника."""
        self._check_open()
        return self

    @property
    def doc_lengths(self):
        self._check_open()
        return MappingProxyType(self._doc_lengths)

    @property
    def doc_meta(self):
        self._check_open()
        return MappingProxyType(self._doc_meta)

    @property
    def texts(self):
        self._check_open()
        return MappingProxyType(self._texts)

    @property
    def has_positions(self):
        self._check_open()
        return self._positions is not None

    def positions(self, term, doc_id):
        self._check_open()
        if not self.has_positions:
            raise ValueError("Фразовий пошук потребує побудови з --positions")
        return self._positions.get(term, {}).get(doc_id, ())

    def _evaluate_query(self, query):
        from findex.query import parse

        return frozenset(parse(query).evaluate(self))

    def matched_ids(self, query):
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

    def cache_info(self):
        return self._cached_ids.cache_info()

    def close(self):
        """Звільнити власні буфери й кеш; зовнішні знімки належать викликачеві."""
        self._cached_ids.cache_clear()
        self._postings.clear()
        self._doc_lengths.clear()
        self._doc_meta.clear()
        self._texts.clear()
        if self._positions is not None:
            self._positions.clear()
        self.__dict__.pop("avg_doc_length", None)
        self.closed = True

    def __getstate__(self):
        self._check_open()
        return dict(
            postings=self._postings,
            doc_lengths=self._doc_lengths,
            doc_meta=self._doc_meta,
            storage=self.storage,
            positions=self._positions,
            texts=self._texts,
        )

    def __setstate__(self, state):
        # Старий slotted Index лабораторної 2 мав стан (None, словник).
        if isinstance(state, tuple):
            state = state[1]
        self.__init__(**state)


def pairs(postings: PostingList) -> Iterator[tuple[int, int]]:
    """Однаковий інтерфейс для об'єктів та компактних масивів."""
    if isinstance(postings, ArrayPostings):
        yield from zip(postings.doc_ids, postings.tfs, strict=True)
    else:
        for posting in postings:
            yield posting.doc_id, posting.tf


def document_ids(postings: PostingList) -> list[int]:
    if isinstance(postings, ArrayPostings):
        return list(postings.doc_ids)
    return [posting.doc_id for posting in postings]
