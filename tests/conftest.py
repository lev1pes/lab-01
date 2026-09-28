"""Спільний маленький корпус; кожен тест отримує власні змінні об'єкти."""

import pytest

from findex.corpus import iter_documents
from findex.index import build_index
from findex.store import save


@pytest.fixture
def tiny_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    texts = {
        "a.txt": "Python async event loop. Кіт café.",
        "b.txt": "Python java event loop. Дім.",
        "c.txt": "Python await. Кіт кіт.",
        "empty.txt": "",
    }
    for name, text in texts.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


@pytest.fixture
def tiny_index(tiny_corpus):
    index = build_index(iter_documents(tiny_corpus), positions=True)
    yield index
    index.close()


@pytest.fixture
def index_path(tiny_index, tmp_path):
    path = tmp_path / "index.bin"
    save(tiny_index, path)
    return path
