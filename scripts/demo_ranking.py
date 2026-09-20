"""Демо захисту: контекст, фраза, рейтинг, снипети та повторний запит."""

import argparse
import logging
from pathlib import Path

from findex.scoring import BM25
from findex.search import search
from findex.store import open_index


def main():
    parser = argparse.ArgumentParser(description="Демонстрація лабораторної 3")
    parser.add_argument("path", nargs="?", type=Path, default=Path("index.bin"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    query = 'python AND (async OR await) NOT java "event loop"'
    with open_index(args.path) as index:
        print(repr(index))
        print(f"Середня довжина: {index.avg_doc_length:.2f} токенів")
        print(f"Запит: {query}")
        for result in search(index, query, BM25(), k=5):
            print(result)
            print(f"  {index.doc_meta[result.doc_id].path}")
        search(index, query, BM25(), k=5)
        print(index.cache_info())
    print(f"Індекс закрито: {index.closed}")


if __name__ == "__main__":
    main()
