import json
import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import STORAGES, build_index
from findex.index import main as index_main
from findex.models import DocMeta, Posting, pairs
from findex.search import boolean_search as search
from findex.search import main as search_main
from findex.search import merge_and, merge_not, merge_or
from findex.store import load, save


def documents():
    for name, text in [
        ("a", "Кіт кіт дім café"),
        ("b", "Дім сад"),
        ("c", "Сад море"),
        ("empty", ""),
    ]:
        yield Document(f"{name}.txt", Path(f"{name}.txt"), text)


@pytest.mark.parametrize("storage", STORAGES)
def test_build_single_pass_and_counts(storage):
    class Once:
        used = False

        def __iter__(self):
            assert not self.used
            self.used = True
            yield from documents()

    index = build_index(Once(), storage)
    assert index.doc_lengths == {0: 4, 1: 2, 2: 2, 3: 0}
    assert index.doc_meta[0] == DocMeta("a.txt", "a")
    assert {term: list(pairs(p)) for term, p in index.postings.items()} == {
        "кіт": [(0, 2)],
        "дім": [(0, 1), (1, 1)],
        "café": [(0, 1)],
        "сад": [(1, 1), (2, 1)],
        "море": [(2, 1)],
    }


def test_documents_are_not_materialized(monkeypatch):
    state = []

    def tracked(text):
        state.append(text)
        yield text

    def source():
        yield Document("a", Path("a"), "first")
        assert state == ["first"]
        yield Document("b", Path("b"), "second")

    monkeypatch.setattr("findex.index.tokenize", tracked)
    assert build_index(source()).doc_lengths == {0: 1, 1: 1}


@pytest.mark.parametrize("record", [Posting(7, 2), DocMeta("a.txt", "Назва")])
def test_records_frozen_slotted_hashable(record):
    assert record in {record}
    assert not hasattr(record, "__dict__")
    field = "tf" if isinstance(record, Posting) else "title"
    with pytest.raises(FrozenInstanceError):
        setattr(record, field, 1)


def test_equal_records_have_equal_hashes():
    assert Posting(3, 4) == Posting(3, 4)
    assert hash(Posting(3, 4)) == hash(Posting(3, 4))
    assert len({DocMeta("a", "b"), DocMeta("a", "b")}) == 1


def test_merges_against_set_oracle():
    rng = random.Random(42)
    for _ in range(200):
        a = sorted(rng.sample(range(40), rng.randrange(41)))
        b = sorted(rng.sample(range(40), rng.randrange(41)))
        assert merge_and(a, b) == sorted(set(a) & set(b))
        assert merge_or(a, b) == sorted(set(a) | set(b))
        assert merge_not(a, b) == sorted(set(a) - set(b))


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize(
    "query, expected",
    [
        ("дім сад", [1]),
        ("дім AND сад", [1]),
        ("дім OR сад", [0, 1, 2]),
        ("сад NOT море", [1]),
        ("NOT сад", [0, 3]),
        ("NOT missing", [0, 1, 2, 3]),
        ("кіт OR NOT сад", [0, 3]),
        ("кіт OR дім сад", [1]),
        ("missing", []),
        ("missing OR море", [2]),
        ("КІТ", [0]),
        ("cafe\u0301", [0]),
        ("NOT NOT кіт", [0]),
        ("кіт кіт", [0]),
        ("кіт NOT кіт", []),
        ("дім AND NOT кіт", [1]),
    ],
)
def test_queries(storage, engine, query, expected):
    assert search(build_index(documents(), storage), query, engine) == expected


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "OR кіт",
        "AND кіт",
        "NOT",
        "кіт OR",
        "кіт AND",
        "кіт OR OR сад",
        "кіт NOT OR сад",
        "!!!",
        "(кіт)",
        '"кіт"',
        "well-known",
    ],
)
def test_invalid_query(query):
    with pytest.raises(ValueError):
        search(build_index(documents()), query)


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("format", ["pickle", "json"])
def test_roundtrip_and_empty(tmp_path, storage, format):
    for source in (documents(), iter(())):
        original = build_index(source, storage)
        path = tmp_path / f"index.{format}"
        save(original, path, format)
        restored = load(path, format)
        assert original == restored
        assert search(restored, "NOT сад") == search(original, "NOT сад")


@pytest.mark.parametrize("damage", ["version", "order", "length", "duplicate", "type"])
def test_invalid_json(tmp_path, damage):
    path = tmp_path / "index.json"
    save(build_index(documents()), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if damage == "version":
        data["version"] = 99
    elif damage == "order":
        data["postings"]["дім"].reverse()
    elif damage == "length":
        data["documents"][0][1] = 100
    elif damage == "duplicate":
        data["documents"].append(data["documents"][0])
    else:
        data["postings"]["кіт"][0][1] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def test_truncated_pickle(tmp_path):
    path = tmp_path / "index.bin"
    path.write_bytes(b"\x80\x05")
    with pytest.raises(ValueError):
        load(path)


@pytest.mark.parametrize("suffix", ["bin", "json"])
def test_cli_saved_search_without_corpus(tmp_path, capsys, suffix):
    corpus = tmp_path / "texts"
    corpus.mkdir()
    text = corpus / "a.txt"
    text.write_text("Кіт кіт", encoding="utf-8")
    target = tmp_path / f"index.{suffix}"
    index_main([str(corpus), "--out", str(target)])
    assert "Пікова пам'ять" in capsys.readouterr().out
    text.unlink()
    search_main([str(target), "кіт", "--engine", "set"])
    output = capsys.readouterr().out
    assert "Знайдено документів: 1" in output
    assert "a.txt" in output
    assert "Загальний час" in output


def test_invalid_cli_arguments(tmp_path):
    with pytest.raises(SystemExit) as error:
        index_main([str(tmp_path / "missing"), "--out", str(tmp_path / "i.bin")])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        search_main([str(tmp_path / "missing.bin"), "word"])
    assert error.value.code == 2
