"""Графік абсолютного прискорення та масштабування кожного інтерпретатора."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data = json.loads(Path("benchmarks/lab05.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    series = [
        ("gil", "serial", "3.12 · serial"),
        ("gil", "threads", "3.12 · threads"),
        ("gil", "processes", "3.12 · processes"),
        ("free", "threads", "3.13t · threads, GIL=False"),
    ]
    counts = sorted({r["workers"] for r in rows})
    for axis, key, title in zip(
        axes,
        ("speedup", "runtime_speedup"),
        ("Відносно serial Python 3.12", "Відносно serial свого інтерпретатора"),
        strict=True,
    ):
        axis.plot(counts, counts, "--", color="#999999", label="Ідеал: S(N) = N")
        for runtime, executor, label in series:
            selected = [
                r for r in rows if r["runtime"] == runtime and r["executor"] == executor
            ]
            axis.plot(
                [r["workers"] for r in selected],
                [r[key] for r in selected],
                marker="o",
                label=label,
                linewidth=2,
            )
        axis.axhline(1, color="#bbbbbb", linewidth=0.8)
        axis.set(
            xlabel="Кількість воркерів",
            ylabel="Прискорення, рази",
            title=title,
            xticks=counts,
            ylim=(0, max(counts) + 0.4),
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "findex: 503 документи, позиції токенів; медіани трьох повторів", fontsize=12
    )
    fig.savefig("benchmarks/lab05-speedup.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
