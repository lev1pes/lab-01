"""Читати каталог текстів у UTF-8 по одному документу."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Видавати тексти .txt; пропускати помилкові файли з попередженням."""
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
