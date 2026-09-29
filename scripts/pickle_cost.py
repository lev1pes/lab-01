"""Окремо від основної таблиці: ціна серіалізації аргументів і partial."""

import json
import pickle
import statistics
import time
from pathlib import Path

from findex.parallel import build_partial, document_paths


def measure(value):
    saves, loads = [], []
    for _ in range(3):
        started = time.perf_counter()
        payload = pickle.dumps(value, protocol=5)
        saves.append(time.perf_counter() - started)
        started = time.perf_counter()
        restored = pickle.loads(payload)
        loads.append(time.perf_counter() - started)
        del restored
    return dict(
        bytes=len(payload),
        dumps_seconds=statistics.median(saves),
        loads_seconds=statistics.median(loads),
    )


if __name__ == "__main__":
    root = Path("data/python-docs")
    paths = document_paths(root)
    partial = build_partial(paths, root, positions=True)
    result = {
        "paths_and_options": measure((paths, root, 0, "slots", True)),
        "partial_result": measure(partial),
        "repeats": 3,
    }
    Path("benchmarks/lab05-pickle.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
