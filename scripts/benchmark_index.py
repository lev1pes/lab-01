"""Відтворювані заміри лабораторної 2; кожна структура в окремому процесі."""

import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import timeit
import tracemalloc
from array import array
from datetime import UTC, datetime
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import STORAGES, build_index
from findex.models import Index, pairs
from findex.search import boolean_search as search
from findex.store import load, save


def fingerprint(index: Index) -> str:
    """Порівнюємо весь вміст, незалежно від представлення постінгів."""
    digest = hashlib.sha256()
    for doc_id, meta in sorted(index.doc_meta.items()):
        digest.update(
            json.dumps(
                [doc_id, index.doc_lengths[doc_id], meta.path, meta.title],
                ensure_ascii=True,
            ).encode()
        )
    for term, postings in sorted(index.postings.items()):
        digest.update(
            json.dumps([term, list(pairs(postings))], ensure_ascii=True).encode()
        )
    return digest.hexdigest()


def measure_storage(root: Path, directory: Path, storage: str) -> dict:
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    index = build_index(iter_documents(root), storage)
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "storage": storage,
        "build_seconds": elapsed,
        "current_bytes": current,
        "peak_bytes": peak,
        "documents": len(index.doc_meta),
        "tokens": sum(index.doc_lengths.values()),
        "vocabulary": len(index.postings),
        "postings": sum(map(len, index.postings.values())),
        "fingerprint": fingerprint(index),
        "formats": [],
    }
    for format in ("pickle", "json"):
        path = directory / f"{storage}.{format}"
        start = time.perf_counter()
        save(index, path, format)
        result["formats"].append(
            {
                "format": format,
                "bytes": path.stat().st_size,
                "save_seconds": time.perf_counter() - start,
            }
        )
    del index
    gc.collect()
    for row in result["formats"]:
        start = time.perf_counter()
        restored = load(directory / f"{storage}.{row['format']}", row["format"])
        row["load_seconds"] = time.perf_counter() - start
        assert fingerprint(restored) == result["fingerprint"]
        del restored
        gc.collect()
    return result


def measure_queries(index: Index) -> dict:
    frequent = sorted(index.postings, key=lambda t: (-len(index.postings[t]), t))[:2]
    rare = sorted(index.postings, key=lambda t: (len(index.postings[t]), t))[:2]
    if len(frequent) < 2:
        raise ValueError("Для порівняння потрібно хоча б два різні терміни")
    rows = []
    for label, terms in (("frequent", frequent), ("rare", rare)):
        for operator in ("AND", "OR"):
            query = f"{terms[0]} {operator} {terms[1]}"
            expected = search(index, query, "merge")
            assert search(index, query, "set") == expected
            row = {"group": label, "query": query, "matches": len(expected)}
            for engine in ("merge", "set"):
                runs = timeit.repeat(
                    lambda engine=engine, query=query: search(index, query, engine),
                    repeat=7,
                    number=200,
                )
                row[f"{engine}_microseconds"] = statistics.median(runs) / 200 * 1e6
                row[f"{engine}_batch_seconds"] = runs
            rows.append(row)
    return {
        "term_selection": "document_frequency_then_lexicographic",
        "terms": [{"term": t, "df": len(index.postings[t])} for t in frequent + rare],
        "repeat": 7,
        "number": 200,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Заміри пам'яті, форматів та пошуку")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab02.json"))
    parser.add_argument("--directory", type=Path, default=Path("artifacts/lab02"))
    parser.add_argument("--worker", choices=STORAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        print(json.dumps(measure_storage(args.root, args.directory, args.worker)))
        return
    rows = []
    for storage in STORAGES:
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.root),
                "--directory",
                str(args.directory),
                "--worker",
                storage,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rows.append(json.loads(process.stdout))
        print(f"Виміряно: {storage}", flush=True)
    assert len({row["fingerprint"] for row in rows}) == 1
    report = {
        "measured_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "array_itemsize": array("I").itemsize,
        "corpus": args.root.as_posix(),
        "storage": rows,
        "search": measure_queries(load(args.directory / "slots.pickle")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(f"Результати: {args.out}")


if __name__ == "__main__":
    main()
