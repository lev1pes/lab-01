import inspect
import logging
from pathlib import Path

import pytest

from findex.corpus import iter_documents


def test_recursive_documents_and_bad_encoding(tmp_path, caplog):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.txt").write_text("Привіт", encoding="utf-8-sig")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "skip.md").write_text("ignored", encoding="utf-8")
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfeinvalid")
    with caplog.at_level(logging.WARNING):
        documents = {doc.doc_id: doc for doc in iter_documents(tmp_path)}
    assert set(documents) == {"nested/a.txt", "empty.txt"}
    assert documents["nested/a.txt"].text == "Привіт"
    assert documents["empty.txt"].text == ""
    assert "broken.txt" in caplog.text


def test_only_requested_file_is_opened(tmp_path, monkeypatch):
    for number in range(3):
        (tmp_path / f"{number}.txt").write_text("text", encoding="utf-8")
    opened = []
    original_open = Path.open

    def track_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    assert inspect.isgeneratorfunction(iter_documents)
    documents = iter_documents(tmp_path)
    assert opened == []
    first = next(documents)
    assert opened == [first.path]
    documents.close()


def test_missing_root_raises_on_consumption(tmp_path):
    documents = iter_documents(tmp_path / "missing")
    with pytest.raises(NotADirectoryError):
        next(documents)


def test_unreadable_file_is_skipped(tmp_path, monkeypatch, caplog):
    (tmp_path / "blocked.txt").touch()

    def denied(*args, **kwargs):
        raise PermissionError("тестова помилка доступу")

    monkeypatch.setattr(Path, "open", denied)
    assert list(iter_documents(tmp_path)) == []
    assert "тестова помилка доступу" in caplog.text
