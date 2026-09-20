"""Позиції, три контрольні перевірки рейтингу й P@5 за фіксованими мітками."""

import argparse
import gc
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.models import Posting
from findex.scoring import BM25, TfIdf
from findex.search import search
from findex.store import open_index, save


def measure_build(root, path, positions):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    index = build_index(iter_documents(root), positions=positions)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    save(index, path)
    report = {
        "positions": positions,
        "seconds": elapsed,
        "peak_bytes": peak,
        "file_bytes": path.stat().st_size,
        "documents": index.num_docs,
        "tokens": sum(index.doc_lengths.values()),
        "terms": len(index),
        "postings": sum(index.df(term) for term in index),
    }
    index.close()
    return report


def sanity(index):
    scorer = BM25()
    rare_term, common_term = "dataclasses", "the"
    ranked = search(index, f"{rare_term} OR {common_term}", scorer, index.num_docs)
    rare_docs = {p.doc_id for p in index[rare_term]}
    rare = next(r for r in ranked if r.doc_id in rare_docs)
    common = next(r for r in ranked if r.doc_id not in rare_docs)
    assert rare.score > common.score
    rarity = {
        "query": f"{rare_term} OR {common_term}",
        "rare_df": index.df(rare_term),
        "common_df": index.df(common_term),
        "rare_path": index.doc_meta[rare.doc_id].path,
        "rare_score": rare.score,
        "common_only_path": index.doc_meta[common.doc_id].path,
        "common_only_score": common.score,
    }
    # Контрольований розрахунок: реальний документ і df, змінюється тільки tf.
    chosen = min(
        (p for p in index["python"] if p.tf >= 20),
        key=lambda p: abs(index.doc_length(p.doc_id) - index.avg_doc_length),
    )
    values = {
        str(tf): scorer.score("python", Posting(chosen.doc_id, tf), index)
        for tf in (1, 2, 19, 20)
    }
    ratio = (values["20"] - values["19"]) / (values["2"] - values["1"])
    assert 0 < ratio < 0.03
    saturation = {
        "term": "python",
        "path": index.doc_meta[chosen.doc_id].path,
        "actual_tf": chosen.tf,
        "length": index.doc_length(chosen.doc_id),
        "scores_at_controlled_tf": values,
        "marginal_gain_ratio": ratio,
    }
    once = [p for p in index["generator"] if p.tf == 1]
    short = min(once, key=lambda p: index.doc_length(p.doc_id))
    long = max(once, key=lambda p: index.doc_length(p.doc_id))
    short_score = scorer.score("generator", short, index)
    long_score = scorer.score("generator", long, index)
    assert short_score > long_score
    lengths = {
        "term": "generator",
        "tf": 1,
        "short_path": index.doc_meta[short.doc_id].path,
        "short_length": index.doc_length(short.doc_id),
        "short_score": short_score,
        "long_path": index.doc_meta[long.doc_id].path,
        "long_length": index.doc_length(long.doc_id),
        "long_score": long_score,
    }
    return {"rarity": rarity, "saturation": saturation, "length": lengths}


def evaluate(index, judgments):
    known_paths = {meta.path for meta in index.doc_meta.values()}
    rows = []
    for case in judgments["queries"]:
        relevant = set(case["relevant"])
        assert relevant <= known_paths, relevant - known_paths
        row = {"query": case["query"], "intent": case["intent"]}
        for name, scorer in (("tfidf", TfIdf()), ("bm25", BM25())):
            results = search(index, case["query"], scorer, k=5)
            paths = [index.doc_meta[r.doc_id].path for r in results]
            row[name] = {
                "precision_at_5": len(set(paths) & relevant) / 5,
                "top_5": [
                    {
                        "path": index.doc_meta[r.doc_id].path,
                        "score": r.score,
                        "snippet": r.snippet,
                        "relevant": index.doc_meta[r.doc_id].path in relevant,
                    }
                    for r in results
                ],
            }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Перевірки якості ранжування")
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--judgments", type=Path, default=Path("benchmarks/judgments.json")
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab03.json"))
    parser.add_argument("--directory", type=Path, default=Path("artifacts/lab03"))
    parser.add_argument(
        "--worker", choices=("plain", "positions"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        row = measure_build(
            args.root, args.directory / f"{args.worker}.bin", args.worker == "positions"
        )
        print(json.dumps(row))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_name("lab03-trace.txt")
    rows, logs = [], []
    for mode in ("plain", "positions"):
        run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.root),
                "--directory",
                str(args.directory),
                "--worker",
                mode,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        rows.append(json.loads(run.stdout))
        logs.append(run.stderr)
        print(f"Виміряно: {mode}", flush=True)
    trace.write_text("".join(logs), encoding="utf-8")
    handler = logging.FileHandler(trace, mode="a", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[handler])
    judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
    with open_index(args.directory / "positions.bin") as index:
        checks = sanity(index)
        evaluation = evaluate(index, judgments)
        query = 'python AND (async OR await) NOT java "event loop"'
        first = search(index, query, k=5)
        second = search(index, query, k=5)
        assert first == second
        demo = [
            {
                "path": index.doc_meta[r.doc_id].path,
                "score": r.score,
                "snippet": r.snippet,
            }
            for r in first
        ]
        report = {
            "measured_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "corpus": args.root.as_posix(),
            "builds": rows,
            "judgments_sha256": hashlib.sha256(args.judgments.read_bytes()).hexdigest(),
            "sanity": checks,
            "evaluation": evaluation,
            "mean_precision_at_5": {
                name: sum(r[name]["precision_at_5"] for r in evaluation)
                / len(evaluation)
                for name in ("tfidf", "bm25")
            },
            "demo_query": query,
            "demo_results": demo,
            "cache_info": index.cache_info()._asdict(),
        }
    report["closed_after_with"] = index.closed
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Результати: {args.out}")


if __name__ == "__main__":
    main()
