"""Векторні формули та стабільний top-k без повного сортування."""

import heapq

import numpy as np
from numpy.typing import NDArray

from findex.models import Index
from findex.query import parse
from findex.scoring import BM25, TfIdf


def top_indices(
    scores: NDArray[np.float64], ids: NDArray[np.int32], k: int
) -> NDArray[np.intp]:
    """На межі k однакові бали розв'язуються меншим doc_id, як у лабі 3."""
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    split = np.argpartition(scores, len(scores) - k)[-k:]
    threshold = scores[split].min()
    greater = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    tied = tied[np.argsort(ids[tied], kind="stable")[: k - len(greater)]]
    chosen = np.concatenate((greater, tied))
    return chosen[np.lexsort((ids[chosen], -scores[chosen]))]


def rank_numpy(
    index: Index, query: str, scorer: BM25 | TfIdf, k: int
) -> list[tuple[int, float]]:
    candidates = np.array(sorted(index.matched_ids(query)), dtype=np.int32)
    scores = np.zeros(index.num_docs, dtype=np.float64)
    for term in sorted(parse(query).terms()):
        postings = index.numpy_postings.get(term)
        if postings is None:
            continue
        ids, tfs = postings.doc_ids, postings.tfs.astype(np.float64)
        df = len(ids)
        if isinstance(scorer, BM25):
            if not index.avg_doc_length:
                continue
            idf = np.log1p((index.num_docs - df + 0.5) / (df + 0.5))
            norm = scorer.k1 * (
                1 - scorer.b + scorer.b * index.length_array[ids] / index.avg_doc_length
            )
            values = idf * tfs * (scorer.k1 + 1) / (tfs + norm)
        else:
            values = (1 + np.log(tfs)) * np.log1p(index.num_docs / df)
        scores[ids] += values
    selected = top_indices(scores[candidates], candidates, k)
    return [(int(candidates[i]), float(scores[candidates[i]])) for i in selected]


def rank_python(
    index: Index, query: str, scorer: BM25 | TfIdf, k: int
) -> list[tuple[int, float]]:
    """Формула і heapq лабораторної 3, без снипетів для чесного порівняння."""
    candidates = index.matched_ids(query)
    scores = dict.fromkeys(candidates, 0.0)
    for term in sorted(parse(query).terms()):
        for posting in index.get(term, ()):
            if posting.doc_id in candidates:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    return heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
