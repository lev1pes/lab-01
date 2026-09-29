"""Ранжований пошук із новим парсером і сумісний булевий режим лаби 2."""

import argparse
import heapq
import logging
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from findex.models import Index, document_ids
from findex.query import parse
from findex.scoring import BM25, Scorer, TfIdf
from findex.snippets import snippet
from findex.store import open_index
from findex.timing import timed
from findex.tokenize import tokenize

type Engine = Literal["merge", "set"]


def merge_and(a: list[int], b: list[int]) -> list[int]:
    result: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def merge_or(a: list[int], b: list[int]) -> list[int]:
    result: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def merge_not(a: list[int], b: list[int]) -> list[int]:
    """Різниця a мінус b без побудови множин."""
    result: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            j += 1
    result.extend(a[i:])
    return result


def parse_query(query: str) -> list[tuple[str, bool, str]]:
    """Оператори лише великими літерами; дужки та фрази не підтримуються."""
    if any(character in query for character in '()"'):
        raise ValueError("Дужки та фразовий пошук поки не підтримуються")
    words = query.split()
    if not words:
        raise ValueError("Порожній запит")
    parts: list[tuple[str, bool, str]] = []
    i = 0
    while i < len(words):
        operation = "AND"
        if words[i] in ("AND", "OR"):
            if not parts:
                raise ValueError("Запит має починатися з терміна або NOT")
            operation = words[i]
            i += 1
        negative = False
        while i < len(words) and words[i] == "NOT":
            negative = not negative
            i += 1
        if i == len(words) or words[i] in ("AND", "OR"):
            raise ValueError("Після оператора потрібен термін")
        terms = list(tokenize(words[i]))
        if len(terms) != 1:
            raise ValueError("Кожен операнд має утворювати рівно один токен")
        parts.append((operation, negative, terms[0]))
        i += 1
    return parts


def boolean_search(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    if engine not in ("merge", "set"):
        raise ValueError(f"Невідомий рушій: {engine}")
    parts = parse_query(query)
    if engine == "set":
        selected: set[int] | None = None
        for operation, negative, term in parts:
            operand = set(document_ids(index.get(term, ())))
            if negative:
                operand = set(index.doc_meta) - operand
            if selected is None:
                selected = operand
            elif operation == "OR":
                selected |= operand
            else:
                selected &= operand
        return sorted(selected or ())
    result: list[int] | None = None
    for operation, negative, term in parts:
        ids = document_ids(index.get(term, ()))
        if negative and result is not None and operation == "AND":
            result = merge_not(result, ids)
            continue
        if negative:
            ids = merge_not(sorted(index.doc_meta), ids)
        if result is None:
            result = ids
        elif operation == "OR":
            result = merge_or(result, ids)
        else:
            result = merge_and(result, ids)
    return result or []


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    doc_id: int = field(compare=False)
    score: float
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)
    _tie: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tie", -self.doc_id)

    def __str__(self) -> str:
        return f"{self.score:.5f}\t{self.doc_id}\t{self.title}\n  {self.snippet}"


@timed
def search_reference(
    index: Index, query: str, scorer: Scorer | None = None, k: int = 10
) -> list[SearchResult]:
    """Ранжувати булеву відповідь; NOT фільтрує, але не додає позитивних балів."""
    if type(k) is not int or k < 0:
        raise ValueError("k має бути невід'ємним цілим числом")
    scorer = BM25() if scorer is None else scorer
    candidates = index.matched_ids(query)
    terms = parse(query).terms()
    scores = dict.fromkeys(candidates, 0.0)
    for term in sorted(terms):
        for posting in index.get(term, ()):
            if posting.doc_id in candidates:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    top = heapq.nlargest(
        k,
        (
            SearchResult(doc, score, index.doc_meta[doc].title)
            for doc, score in scores.items()
        ),
    )
    return [
        replace(result, snippet=snippet(index.texts.get(result.doc_id, ""), terms))
        for result in top
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ранжований пошук у збереженому індексі"
    )
    parser.add_argument("index", type=Path, help="довірений файл індексу")
    parser.add_argument("query", help="терміни та оператори AND, OR, NOT")
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default=None,
        help="старий булевий режим лабораторної 2",
    )
    parser.add_argument("--format", choices=("pickle", "json"), default=None)
    parser.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("--top", type=int, default=10, help="кількість результатів")
    parser.add_argument("--boolean", action="store_true", help="режим лабораторної 2")
    parser.add_argument(
        "--repeat", type=int, default=1, help="повторити запит в одному індексі"
    )
    parser.add_argument("--verbose", action="store_true", help="журнал часу та кешу")
    args = parser.parse_args(argv)
    if __name__ == "__main__":
        from findex.cli import configure_logging

        configure_logging(2 if getattr(args, "verbose", False) else 1)
    args.boolean = args.boolean or args.engine is not None
    args.engine = args.engine or "merge"
    if args.top < 0 or args.repeat < 1:
        parser.error("--top має бути >= 0, --repeat має бути >= 1")
    tracemalloc.start()
    started = time.perf_counter()
    try:
        with open_index(args.index, args.format) as index:
            loaded = time.perf_counter()
            scorer = BM25() if args.scorer == "bm25" else TfIdf()
            results: list[int] | list[SearchResult] = []
            for _ in range(args.repeat):
                results = (
                    boolean_search(index, args.query, args.engine)
                    if args.boolean
                    else search(index, args.query, scorer, args.top)
                )
            finished = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            lines: list[str] = []
            for result in results:
                doc = result if isinstance(result, int) else result.doc_id
                meta = index.doc_meta[doc]
                lines.append(
                    f"{doc}\t{meta.title}\t{meta.path}"
                    if args.boolean
                    else f"{result}\n  {meta.path}"
                )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    finally:
        tracemalloc.stop()
    print(f"Знайдено документів: {len(results)}")
    for line in lines:
        print(line)
    log = logging.getLogger("findex")
    log.info("Завантаження: %.6f с", loaded - started)
    log.info("Пошук: %.6f с", finished - loaded)
    log.info("Загальний час: %.6f с", finished - started)
    log.info("Пікова пам'ять: %.3f МіБ", peak / 1024**2)


@timed
def search(
    index: Index, query: str, scorer: Scorer | None = None, k: int = 10
) -> list[SearchResult]:
    """NumPy для стандартних скорерів; duck-typed скорер має старий шлях."""
    from findex.vector import rank_numpy

    if type(k) is not int or k < 0:
        raise ValueError("k має бути невід'ємним цілим числом")
    algorithm = BM25() if scorer is None else scorer
    if not isinstance(algorithm, (BM25, TfIdf)):
        return search_reference(index, query, algorithm, k)
    terms = parse(query).terms()
    return [
        SearchResult(
            doc,
            score,
            index.doc_meta[doc].title,
            snippet(index.texts.get(doc, ""), terms),
        )
        for doc, score in rank_numpy(index, query, algorithm, k)
    ]


if __name__ == "__main__":
    main()
