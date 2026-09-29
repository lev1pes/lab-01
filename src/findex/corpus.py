"""Читати каталог текстів у UTF-8 по одному документу."""

import json
import logging
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import NamedTuple, cast
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str
    title: str = ""


def iter_documents(root: Path) -> Generator[Document, None, None]:
    """Видавати тексти .txt; пропускати помилкові файли з попередженням."""
    if root.is_file() and root.suffix == ".jsonl":
        yield from iter_jsonl(root)
        return
    if not root.is_dir():
        raise NotADirectoryError(f"Каталог корпусу не існує: {root}")

    for path in root.rglob("*.txt"):
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8-sig") as source:
                text = source.read()
        except (UnicodeError, OSError) as error:
            logger.warning("Пропущено %s: %s", path, error)
            continue

        yield Document(path.relative_to(root).as_posix(), path, text)


def iter_jsonl(path: Path) -> Iterator[Document]:
    """Один JSON-об'єкт на рядок; помилка містить номер рядка."""
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw: object = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("Потрібен JSON-об'єкт")
                record = cast(dict[str, object], raw)
                if not all(
                    isinstance(record.get(key), str) for key in ("url", "title", "text")
                ):
                    raise ValueError("Потрібні рядки url, title, text")
                url, title, text = (
                    str(record["url"]),
                    str(record["title"]),
                    str(record["text"]),
                )
                yield Document(url, Path(urlsplit(url).path), text, title)
            except (ValueError, TypeError) as error:
                raise ValueError(f"JSONL {path}, рядок {number}: {error}") from error
