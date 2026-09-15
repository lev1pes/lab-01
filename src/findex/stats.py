"""Запуск: python -m findex.stats data/ [--limit N] [--mode eager]."""

import argparse
import json
import logging
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import islice
from pathlib import Path
from typing import NamedTuple

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


class Stats(NamedTuple):
    documents: int
    tokens: int
    counts: Counter[str]


def collect_lazy(documents: Iterable[Document]) -> Stats:
    """Зберігати лічильник термінів і поточні документи та токени."""
    counts: Counter[str] = Counter()
    document_count = 0
    for document in documents:
        document_count += 1
        counts.update(tokenize(document.text))
    return Stats(document_count, counts.total(), counts)


def collect_eager(documents: Iterable[Document]) -> Stats:
    """Навмисно зберігати всі тексти та списки токенів до кінця підрахунку."""
    all_documents = list(documents)
    token_lists = [list(tokenize(document.text)) for document in all_documents]
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return Stats(len(all_documents), counts.total(), counts)


def non_negative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("ліміт має бути невід’ємним")
    return number


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Підрахунок документів і токенів")
    parser.add_argument("root", type=Path, help="каталог із текстами .txt")
    parser.add_argument(
        "--limit", type=non_negative, help="максимальна кількість документів"
    )
    parser.add_argument(
        "--mode",
        choices=("lazy", "eager"),
        default="lazy",
        help="спосіб обробки корпусу",
    )
    parser.add_argument(
        "--json", action="store_true", help="вивести результат у форматі JSON"
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"каталог корпусу не існує: {args.root}")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    collector = collect_lazy if args.mode == "lazy" else collect_eager
    tracemalloc.start()
    started = time.perf_counter()
    try:
        documents = iter_documents(args.root)
        if args.limit is not None:
            documents = islice(documents, args.limit)
        result = collector(documents)
        # За однакової частоти впорядковуємо слова за абеткою.
        top = sorted(result.counts.items(), key=lambda item: (-item[1], item[0]))[:50]
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    report = {
        "mode": args.mode,
        "documents": result.documents,
        "tokens": result.tokens,
        "vocabulary": len(result.counts),
        "top_50": top,
        "elapsed_seconds": elapsed,
        "peak_bytes": peak,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(f"Режим: {args.mode}")
        print(f"Документів: {result.documents}")
        print(f"Токенів: {result.tokens}")
        print(f"Розмір словника: {len(result.counts)}")
        print(f"Час: {elapsed:.3f} с")
        print(f"Пікова пам’ять: {peak / 1024**2:.3f} МіБ")
        print("Топ-50 термінів:")
        for term, count in top:
            print(f"{term}\t{count}")


if __name__ == "__main__":
    main()
