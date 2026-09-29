"""Повторюваний гарячий шлях без імпорту CLI та дискового load у циклі."""

import argparse
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.search import search
from findex.store import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["search", "index"])
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if args.operation == "search":
        index = load(Path("data/web-index.json"))
        for _ in range(args.repeat):
            search(index, "python async await")
        index.close()
    else:
        for _ in range(args.repeat):
            build_index(
                iter_documents(Path("data/python-docs")), positions=True
            ).close()
