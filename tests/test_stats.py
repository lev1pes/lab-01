import json
from collections import Counter
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.stats import collect_eager, collect_lazy, main


def sample_documents():
    yield Document("a", Path("a.txt"), "Python python café")
    yield Document("b", Path("b.txt"), "cafe\u0301 Привіт")
    yield Document("empty", Path("empty.txt"), "")


def test_both_versions_have_identical_full_counts():
    lazy = collect_lazy(sample_documents())
    eager = collect_eager(sample_documents())
    assert lazy == eager
    assert lazy.documents == 3
    assert lazy.tokens == 5
    assert lazy.counts == Counter({"python": 2, "café": 2, "привіт": 1})


@pytest.mark.parametrize("collector", [collect_lazy, collect_eager])
def test_empty_input(collector):
    assert collector(iter(())) == (0, 0, Counter())


def test_limit_does_not_request_an_extra_document(tmp_path, monkeypatch, capsys):
    def guarded_documents(root):
        yield Document("a", root / "a.txt", "one two")
        raise AssertionError("Читання понад ліміт")

    monkeypatch.setattr("findex.stats.iter_documents", guarded_documents)
    main([str(tmp_path), "--limit", "1", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result["documents"] == 1
    assert result["tokens"] == 2


def test_zero_limit(tmp_path, capsys):
    (tmp_path / "a.txt").write_text("word", encoding="utf-8")
    main([str(tmp_path), "--limit", "0", "--json"])
    assert json.loads(capsys.readouterr().out)["documents"] == 0


@pytest.mark.parametrize("limit", ["-1", "abc"])
def test_invalid_limit(tmp_path, limit):
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "--limit", limit])
    assert error.value.code == 2


def test_missing_directory(tmp_path):
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing")])
    assert error.value.code == 2


def test_top_50_and_alphabetical_ties(tmp_path, capsys):
    (tmp_path / "a.txt").write_text(
        " ".join(f"term{i:02}" for i in reversed(range(60))), encoding="utf-8"
    )
    main([str(tmp_path), "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result["vocabulary"] == 60
    assert result["top_50"] == [[f"term{i:02}", 1] for i in range(50)]
    assert result["peak_bytes"] > 0
