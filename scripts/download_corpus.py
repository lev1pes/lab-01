"""Завантажити документацію фіксованої версії Python як тексти .txt у UTF-8."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

VERSION = "3.12.10"
URL = f"https://www.python.org/ftp/python/{VERSION}/Python-{VERSION}.tar.xz"
SHA256 = "07ab697474595e06f06647417d3c7fa97ded07afc1a7e4454c5639919b46eaea"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/python-docs"),
        help="порожній каталог для корпусу",
    )
    args = parser.parse_args()
    output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} не порожній; виберіть порожній каталог")
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "source.tar.xz"
    print(f"Завантаження {URL}")
    with urllib.request.urlopen(URL, timeout=60) as response:
        with archive_path.open("wb") as target:
            shutil.copyfileobj(response, target)
    with archive_path.open("rb") as source:
        actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if actual_hash != SHA256:
        raise ValueError("Контрольна сума архіву не збігається")

    documents = 0
    total_bytes = 0
    prefix = f"Python-{VERSION}/Doc/"
    # Читаємо архів послідовно; шляхи вихідних файлів перевіряємо окремо.
    with tarfile.open(archive_path, "r|xz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name == f"Python-{VERSION}/LICENSE":
                with archive.extractfile(member) as source:
                    (output / "LICENSE").write_bytes(source.read())
            if not member.name.startswith(prefix) or not member.name.endswith(".rst"):
                continue
            relative = PurePosixPath(member.name.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Небезпечний шлях в архіві: {relative}")
            path = output.joinpath(*relative.parts).with_suffix(".txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source:
                text = source.read().decode("utf-8")
            size = path.write_bytes(text.encode("utf-8"))
            total_bytes += size
            documents += 1
    archive_path.unlink()
    manifest = {
        "source": URL,
        "sha256": SHA256,
        "documents": documents,
        "bytes": total_bytes,
        "format": "один файл Doc/**/*.rst на документ .txt; розмітку RST збережено",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Збережено {documents} документів, {total_bytes / 1024**2:.2f} МіБ у {output}"
    )


if __name__ == "__main__":
    main()
