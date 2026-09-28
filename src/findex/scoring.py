"""Два взаємозамінні способи оцінки без спільного базового класу."""

import math
from dataclasses import dataclass
from typing import Protocol

from findex.models import Index, Posting


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        if posting.tf <= 0 or not index.df(term):
            return 0.0
        return (1 + math.log(posting.tf)) * math.log1p(index.num_docs / index.df(term))

    __call__ = score


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 має бути скінченним додатним числом")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b має бути в межах [0, 1]")

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if posting.tf <= 0 or not df or not index.avg_doc_length:
            return 0.0
        idf = math.log1p((index.num_docs - df + 0.5) / (df + 0.5))
        length = index.doc_length(posting.doc_id) / index.avg_doc_length
        denominator = posting.tf + self.k1 * (1 - self.b + self.b * length)
        return idf * posting.tf * (self.k1 + 1) / denominator

    __call__ = score
