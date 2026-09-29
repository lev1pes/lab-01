"""Окремий процес на замір; медіани, сирі повтори та контроль результату."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import psutil

from findex.parallel import BuildMetrics, build_parallel, document_paths
from findex.store import save


class Resources:
    """Windows: збережені HANDLE дають CPU навіть після завершення дочірніх PID."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.handles = {}
        if os.name == "nt":
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
                ctypes.POINTER(wintypes.FILETIME)
            ] * 4
            self.kernel.GetProcessTimes.restype = wintypes.BOOL
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            if os.name == "nt":
                for child in self.process.children(recursive=True):
                    if child.pid not in self.handles:
                        handle = self.kernel.OpenProcess(0x1000, False, child.pid)
                        if handle:
                            self.handles[child.pid] = handle
            self.stop.wait(0.005)

    def finish(self):
        self.stop.set()
        self.thread.join()
        if os.name == "nt":
            cpu = 0.0
            for handle in self.handles.values():
                fields = [wintypes.FILETIME() for _ in range(4)]
                if not self.kernel.GetProcessTimes(
                    handle, *(ctypes.byref(f) for f in fields)
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                cpu += (
                    sum((f.dwHighDateTime << 32) + f.dwLowDateTime for f in fields[2:])
                    / 1e7
                )
                self.kernel.CloseHandle(handle)
            # PeakWorkingSetSize — високий рівень RSS за життя цього свіжого процесу.
            self.peak = self.process.memory_info().peak_wset
            return cpu
        import resource

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime + usage.ru_stime


def one(args):
    # Перевірка після імпорту всіх залежностей: розширення може увімкнути GIL.
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    monitor = Resources()
    metrics = BuildMetrics()
    monitor.thread.start()
    cpu_start = time.process_time()
    started = time.perf_counter()
    paths = document_paths(args.corpus)
    index = build_parallel(
        paths,
        args.corpus,
        workers=args.workers,
        executor=args.executor,
        positions=True,
        metrics=metrics,
    )
    wall = time.perf_counter() - started
    parent_cpu = time.process_time() - cpu_start
    child_cpu = monitor.finish()
    # Серіалізація й хеш — поза інтервалом індексації та виміром RSS.
    target = args.out.with_suffix(".index.json")
    save(index, target)
    fingerprint = hashlib.sha256(target.read_bytes()).hexdigest()
    target.unlink()
    result = dict(
        python=sys.version,
        gil_enabled=gil,
        executor=args.executor,
        workers=args.workers,
        wall=wall,
        cpu=parent_cpu + child_cpu,
        parent_cpu=parent_cpu,
        child_cpu=child_cpu,
        peak_rss_mib=monitor.peak / 1024**2,
        merge=metrics.merge_seconds,
        chunks=metrics.chunks,
        documents=index.num_docs,
        tokens=sum(index.doc_lengths.values()),
        sha256=fingerprint,
        child_processes=len(monitor.handles),
    )
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def matrix(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = sorted({1, 2, 4, 8, os.cpu_count() or 1})
    specs = [
        (sys.executable, "gil", ex, n)
        for ex in ("serial", "threads", "processes")
        for n in counts
    ]
    specs += [(str(args.free_python), "free", "serial", 1)]
    specs += [(str(args.free_python), "free", "threads", n) for n in counts]
    # Той самий 3.13t з примусово увімкненим GIL відділяє версію від блокування.
    specs += [
        (str(args.free_python), "free-gil-on", "serial", 1),
        (str(args.free_python), "free-gil-on", "threads", 6),
    ]
    runs = []
    raw = args.out.parent / "lab05-raw"
    raw.mkdir(exist_ok=True)
    for python, runtime, executor, workers in specs:
        for repeat in range(-1, args.repeats):
            output = raw / f"{runtime}-{executor}-{workers}-{repeat}.json"
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            if runtime == "free-gil-on":
                env["PYTHON_GIL"] = "1"
            else:
                env.pop("PYTHON_GIL", None)
            subprocess.run(
                [
                    python,
                    __file__,
                    str(args.corpus),
                    "--one",
                    "--executor",
                    executor,
                    "--workers",
                    str(workers),
                    "--out",
                    str(output),
                ],
                check=True,
                env=env,
                timeout=180,
            )
            item = json.loads(output.read_text(encoding="utf-8"))
            item.update(runtime=runtime, repeat=repeat)
            if repeat >= 0:
                runs.append(item)
            print(
                f"{runtime} {executor} {workers} повтор {repeat}: {item['wall']:.3f} с",
                flush=True,
            )
        # Контрольна точка після кожної клітинки, щоб не втратити завершені виміри.
        args.out.write_text(
            json.dumps({"runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    assert len({r["sha256"] for r in runs}) == 1, "Різні індекси!"
    rows = []
    for _, runtime, executor, workers in specs:
        group = [
            r
            for r in runs
            if (r["runtime"], r["executor"], r["workers"])
            == (runtime, executor, workers)
        ]
        row = dict(
            runtime=runtime,
            executor=executor,
            workers=workers,
            gil_enabled=group[0]["gil_enabled"],
        )
        row.update(
            {
                key: statistics.median(r[key] for r in group)
                for key in (
                    "wall",
                    "cpu",
                    "parent_cpu",
                    "child_cpu",
                    "peak_rss_mib",
                    "merge",
                )
            }
        )
        row.update(
            wall_min=min(r["wall"] for r in group),
            wall_max=max(r["wall"] for r in group),
        )
        rows.append(row)
    baseline = rows[0]["wall"]
    for row in rows:
        row["speedup"] = baseline / row["wall"]
        own = next(
            r["wall"]
            for r in rows
            if r["runtime"] == row["runtime"]
            and r["executor"] == "serial"
            and r["workers"] == 1
        )
        row["runtime_speedup"] = own / row["wall"]
    machine = dict(
        platform=platform.platform(),
        logical_cpus=os.cpu_count(),
        physical_cores=psutil.cpu_count(logical=False),
        cpu=platform.processor(),
    )
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            machine["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    result = dict(
        machine=machine, repeats=args.repeats, positions=True, rows=rows, runs=runs
    )
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab05.json"))
    parser.add_argument("--free-python", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--one", action="store_true")
    parser.add_argument(
        "--executor", choices=["serial", "threads", "processes"], default="serial"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.one:
        one(args)
    elif args.free_python is None or args.repeats < 3:
        parser.error("Потрібні --free-python і щонайменше три повтори")
    else:
        matrix(args)


if __name__ == "__main__":
    main()
