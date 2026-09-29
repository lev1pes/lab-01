"""Контрольована гонка read-modify-write та виправлення через Lock."""

import dis
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock


def unsafe_increment(counter):
    counter["updates"] += 1


def experiment(locked=False, rounds=100):
    counter = Counter()
    barrier = Barrier(2)
    lock = Lock()

    def worker():
        for _ in range(rounds):
            if locked:
                with lock:
                    unsafe_increment(counter)
            else:
                value = counter["updates"]
                # Навмисно змушуємо обидва потоки прочитати старе значення.
                barrier.wait()
                counter["updates"] = value + 1
                barrier.wait()

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        for future in futures:
            future.result()
    return counter["updates"]


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "python": sys.version,
                "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
                "expected": 200,
                "forced_race": experiment(),
                "with_lock": experiment(True),
            },
            ensure_ascii=False,
        )
    )
    dis.dis(unsafe_increment)
