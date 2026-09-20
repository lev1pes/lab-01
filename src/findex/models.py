"""Структури інвертованого індексу та три способи зберігання постінгів."""

from array import array
from collections.abc import Iterator
from dataclasses import dataclass


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


@dataclass(slots=True)
class Index:
    postings: dict[str, PostingList]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    storage: str = "slots"


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
