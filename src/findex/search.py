"""Булевий пошук: послідовне обчислення AND, OR та NOT."""

import argparse
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path

from findex.models import Index, document_ids
from findex.store import load
from findex.tokenize import tokenize


def merge_and(a: list[int], b: list[int]) -> list[int]:
    result = []
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
    result = []
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
    result = []
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
    parts = []
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


def search(index: Index, query: str, engine: str = "merge") -> list[int]:
    if engine not in ("merge", "set"):
        raise ValueError(f"Невідомий рушій: {engine}")
    result = None
    for operation, negative, term in parse_query(query):
        ids = document_ids(index.postings.get(term, []))
        if engine == "set":
            operand = set(ids)
            if negative:
                operand = set(index.doc_meta) - operand
            if result is None:
                result = operand
            elif operation == "OR":
                result |= operand
            else:
                result &= operand
        else:
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
    return sorted(result) if engine == "set" else result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Булевий пошук у збереженому індексі")
    parser.add_argument("index", type=Path, help="довірений файл індексу")
    parser.add_argument("query", help="терміни та оператори AND, OR, NOT")
    parser.add_argument("--engine", choices=("merge", "set"), default="merge")
    parser.add_argument("--format", choices=("pickle", "json"), default=None)
    args = parser.parse_args(argv)
    tracemalloc.start()
    started = time.perf_counter()
    try:
        index = load(args.index, args.format)
        loaded = time.perf_counter()
        ids = search(index, args.query, args.engine)
        finished = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    finally:
        tracemalloc.stop()
    print(f"Знайдено документів: {len(ids)}")
    for doc_id in ids:
        meta = index.doc_meta[doc_id]
        print(f"{doc_id}\t{meta.title}\t{meta.path}")
    print(f"Завантаження: {loaded - started:.6f} с")
    print(f"Пошук: {finished - loaded:.6f} с")
    print(f"Загальний час: {finished - started:.6f} с")
    print(f"Пікова пам'ять: {peak / 1024**2:.3f} МіБ")


if __name__ == "__main__":
    main()
