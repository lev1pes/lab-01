"""Реальні HTTP-запити: 20 одночасних звернень і oha для трьох режимів."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


async def burst(url):
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        await client.get(url + "/search?q=python")
        await client.get(url + "/__bench/lag?reset=true")
        started = time.perf_counter()
        responses = await asyncio.gather(
            *(client.get(url + "/search?q=python") for _ in range(20))
        )
        wall = time.perf_counter() - started
        assert all(r.status_code == 200 for r in responses)
        lag = (await client.get(url + "/__bench/lag")).json()["max_lag_ms"]
        return {"requests": 20, "wall_seconds": wall, "max_loop_lag_ms": lag}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oha", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("data/web-index.json"))
    args = parser.parse_args()
    dest = Path("benchmarks/lab07")
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for mode, workers in (("async", 1), ("sync", 1), ("sync", 4)):
        name = f"{mode}-{workers}"
        log_path = dest / f"{name}-server.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    f"bench_web_app:make_{mode}",
                    "--factory",
                    "--app-dir",
                    "scripts",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8777",
                    "--workers",
                    str(workers),
                    "--no-access-log",
                ],
                env=dict(
                    os.environ,
                    INDEX_PATH=str(args.index.resolve()),
                    PYTHONIOENCODING="utf-8",
                ),
                stdout=log,
                stderr=log,
            )
            try:
                with httpx.Client(trust_env=False, timeout=2) as client:
                    for _ in range(120):
                        try:
                            if (
                                client.get("http://127.0.0.1:8777/health").status_code
                                == 200
                            ):
                                break
                        except httpx.HTTPError:
                            pass
                        if process.poll() is not None:
                            raise RuntimeError(log_path.read_text(encoding="utf-8"))
                        time.sleep(0.5)
                    else:
                        raise RuntimeError("Сервер не готовий")
                time.sleep(2)
                measured = (
                    asyncio.run(burst("http://127.0.0.1:8777"))
                    if workers == 1
                    else None
                )
                subprocess.run(
                    [
                        str(args.oha.resolve()),
                        "-z",
                        "10s",
                        "-c",
                        "50",
                        "-w",
                        "--output-format",
                        "json",
                        "--no-tui",
                        "-o",
                        str(dest / f"{name}-oha.json"),
                        "http://127.0.0.1:8777/search?q=python&k=10&page=1&scorer=bm25",
                    ],
                    check=True,
                    timeout=120,
                )
                records.append({"mode": mode, "workers": workers, "burst": measured})
                print(name, measured, flush=True)
            finally:
                # Windows: зупинити тільки дерево створеного процесу uvicorn.
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                    )
                else:
                    process.terminate()
                process.wait(timeout=30)
    (dest / "burst.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
