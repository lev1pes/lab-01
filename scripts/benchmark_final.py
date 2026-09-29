"""Медіани scoring-only і P@5 за наперед заданими мітками."""

import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from legacy_snippets import snippet as old_snippet

from findex.models import Index
from findex.scoring import BM25, TfIdf
from findex.search import search, search_reference
from findex.semantic import MiniLM, SemanticIndex, retrieve
from findex.store import load
from findex.vector import rank_numpy, rank_python


def measure(function, repeats=21):
    function()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "repeats": repeats,
    }


if __name__ == "__main__":
    index = load(Path("data/web-index.json"))
    # Слотовий еталон відтворює об'єкти лаби 3; конвертація NumPy поза заміром.
    legacy = Index(
        {t: list(index[t]) for t in index},
        dict(index.doc_lengths),
        dict(index.doc_meta),
        storage="slots",
        positions=index.__getstate__()["positions"],
        texts=dict(index.texts),
    )
    _ = index.numpy_postings, index.length_array
    rows = []
    rare = next(
        t for t in sorted(index) if index.df(t) == 1 and t.isalpha() and len(t) > 5
    )
    for name, scorer in [("bm25", BM25()), ("tfidf", TfIdf())]:
        for query in ["python", rare, "python async await"]:
            a = rank_python(legacy, query, scorer, 10)
            b = rank_numpy(index, query, scorer, 10)
            assert [x[0] for x in a] == [x[0] for x in b]
            old = measure(lambda: rank_python(legacy, query, scorer, 10))
            new = measure(lambda: rank_numpy(index, query, scorer, 10))
            rows.append(
                {
                    "scorer": name,
                    "query": query,
                    "matched": len(index.matched_ids(query)),
                    "python": old,
                    "numpy": new,
                    "speedup": old["median_ms"] / new["median_ms"],
                }
            )
    with patch("findex.search.snippet", old_snippet):
        old_full = measure(lambda: search_reference(legacy, "python async await"), 7)
    new_full = measure(lambda: search(index, "python async await"), 7)
    report = {
        "scoring": rows,
        "full_search": {
            "lab07": old_full,
            "lab08": new_full,
            "speedup": old_full["median_ms"] / new_full["median_ms"],
        },
    }
    Path("benchmarks/lab08/speed.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    semantic = SemanticIndex.load(Path("data/embeddings"), index)
    encoder = MiniLM(cache_dir=Path("../models"))
    evaluation = []
    judgments = json.loads(
        Path("benchmarks/lab08/judgments.json").read_text(encoding="utf-8")
    )
    for case in judgments["queries"]:
        row = {"query": case["query"], "relevant": case["relevant"]}
        for mode in ["keyword", "semantic", "hybrid"]:
            result = retrieve(
                index, case["query"], k=5, mode=mode, semantic=semantic, encoder=encoder
            )
            paths = [index.doc_meta[r.doc_id].path for r in result]
            row[mode] = {
                "p_at_5": len(set(paths) & set(case["relevant"])) / 5,
                "paths": paths,
            }
        evaluation.append(row)
    Path("benchmarks/lab08/evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(
        "P@5",
        {
            mode: statistics.mean(r[mode]["p_at_5"] for r in evaluation)
            for mode in ["keyword", "semantic", "hybrid"]
        },
    )
