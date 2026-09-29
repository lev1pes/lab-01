"""Три повтори для 1/5/20 корутин: власне дзеркало, 200 сторінок, debug=True."""

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter

import psutil
from local_mirror import Mirror

from findex.crawler.fetch import CrawlStats
from findex.crawler.output import collect, crawl_log


def measure(args):
    state = CrawlStats()
    started = perf_counter()
    with crawl_log(args.log), args.out.open("w", encoding="utf-8") as target:
        asyncio.run(
            collect(
                args.seed,
                target,
                stats=state,
                max_pages=200,
                concurrency=args.concurrency,
                per_host=5,
                delay=0.005,
                timeout=5,
            ),
            debug=True,
        )
    wall = perf_counter() - started
    if sys.platform == "win32":
        rss = psutil.Process().memory_info().peak_wset
    else:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
    record = {
        "concurrency": args.concurrency,
        "pages": state.pages,
        "wall": wall,
        "pages_per_second": state.pages / wall,
        "errors": state.errors,
        "peak_rss_mib": rss / 1024**2,
        "requests": state.requests,
        "debug": True,
    }
    # Перевірка змісту поза таймером і після фіксації RSS.
    pages = [
        json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()
    ]
    assert len({page["url"] for page in pages}) == 200
    canonical = sorted((p["url"], p["title"], p["text"]) for p in pages)
    record["content_sha256"] = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert state.pages == 200 and state.in_flight == 0
    print(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--seed", default="http://127.0.0.1:8766/")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/crawl.jsonl"))
    parser.add_argument("--log", type=Path, default=Path("crawl.log"))
    parser.add_argument("--root", type=Path, default=Path("data/python-docs"))
    args = parser.parse_args()
    if args.measure:
        measure(args)
        return
    results = []
    logs = Path("benchmarks/lab06-logs")
    logs.mkdir(parents=True, exist_ok=True)
    with Mirror(args.root) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for repeat in range(3):
                # Чергування порядку зменшує систематичний ефект часу запуску.
                for concurrency in ((1, 5, 20), (20, 1, 5), (5, 20, 1))[repeat]:
                    server.counts.clear()
                    key = f"c{concurrency}-r{repeat}"
                    # CLI дописує журнал; окремий benchmark-прогін починає новий.
                    (logs / f"{key}.log").write_text("", encoding="utf-8")
                    process = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--measure",
                            "--seed",
                            args.seed,
                            "--concurrency",
                            str(concurrency),
                            "--out",
                            f"data/{key}.jsonl",
                            "--log",
                            str(logs / f"{key}.log"),
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=180,
                    )
                    (logs / f"{key}-debug.txt").write_text(
                        process.stderr, encoding="utf-8"
                    )
                    if process.returncode:
                        raise RuntimeError(process.stderr)
                    if (
                        "Executing <" in process.stderr
                        or "never awaited" in process.stderr
                    ):
                        raise RuntimeError(
                            "Asyncio debug виявив блокування; див. журнал"
                        )
                    row = json.loads(process.stdout)
                    row["repeat"] = repeat
                    assert server.counts["/private"] == 0
                    results.append(row)
                    print(key, row, flush=True)
        finally:
            server.shutdown()
            thread.join()
    summaries = []
    for concurrency in (1, 5, 20):
        rows = [r for r in results if r["concurrency"] == concurrency]
        summaries.append(
            {
                "concurrency": concurrency,
                **{
                    k: statistics.median(r[k] for r in rows)
                    for k in (
                        "pages",
                        "wall",
                        "pages_per_second",
                        "errors",
                        "peak_rss_mib",
                        "requests",
                    )
                },
                "wall_min": min(r["wall"] for r in rows),
                "wall_max": max(r["wall"] for r in rows),
            }
        )
    Path("benchmarks/lab06.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "seed": args.seed,
                "max_pages": 200,
                "per_host": 5,
                "delay": 0.005,
                "server_latency": 0.08,
                "repeat_count": 3,
                "rows": summaries,
                "runs": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
