"""Виміряти обидві версії в окремих процесах на тому самому повному корпусі."""

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="каталог корпусу")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results.json"),
        help="файл результатів вимірювання",
    )
    args = parser.parse_args()
    reports = []
    for mode in ("eager", "lazy"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "findex.stats",
                str(args.root),
                "--mode",
                mode,
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        report = json.loads(result.stdout)
        reports.append(report)
        print(
            f"{mode}: {report['documents']} документів, "
            f"{report['peak_bytes'] / 1024**2:.3f} МіБ, "
            f"{report['elapsed_seconds']:.3f} с"
        )
    for field in ("documents", "tokens", "vocabulary", "top_50"):
        if reports[0][field] != reports[1][field]:
            raise ValueError(f"Результати версій відрізняються: {field}")
    payload = {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "corpus": str(args.root),
        "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Результати збережено у {args.output}")


if __name__ == "__main__":
    main()
