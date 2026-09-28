import logging
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.models import Posting
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.scoring import BM25, TfIdf
from findex.search import SearchResult, search
from findex.snippets import snippet
from findex.store import load, open_index, save
from findex.timing import timed


def build(texts, positions=True, storage="slots"):
    return build_index(
        (Document(f"{i}.txt", Path(f"{i}.txt"), text) for i, text in enumerate(texts)),
        storage,
        positions=positions,
    )


def test_mapping_and_cached_property():
    index = build(["a a b", "b", ""])
    assert len(index) == 2 and index.num_docs == 3
    assert "a" in index and "missing" not in index
    assert set(iter(index)) == {"a", "b"}
    assert index["a"] == (Posting(0, 2),)
    assert index.df("a") == 1 and index.df("missing") == 0
    assert index.doc_length(0) == 3
    assert repr(index) == "Index(terms=2, docs=3)"
    assert "avg_doc_length" not in index.__dict__
    assert index.avg_doc_length == 4 / 3
    assert index.__dict__["avg_doc_length"] == 4 / 3
    with pytest.raises(KeyError):
        index["missing"]
    with pytest.raises(TypeError):
        index.doc_lengths[0] = 99
    with pytest.raises(FrozenInstanceError):
        index["a"][0].tf = 99
    assert build([]).avg_doc_length == 0


@pytest.mark.parametrize(
    "query, tree",
    [
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("(a OR b) c", And(Or(Term("a"), Term("b")), Term("c"))),
        ("NOT NOT a", Not(Not(Term("a")))),
        (
            'python AND (async OR await) NOT java "event loop"',
            Term("python")
            & (Term("async") | Term("await"))
            & ~Term("java")
            & Phrase(("event", "loop")),
        ),
        ('"КІТ ДІМ"', Phrase(("кіт", "дім"))),
        ("a AND NOT (b OR c)", Term("a") & ~(Term("b") | Term("c"))),
        ("cafe\u0301", Term("café")),
        ('"a"\nb', And(Phrase(("a",)), Term("b"))),
    ],
)
def test_parser_trees(query, tree):
    assert parse(query) == tree


@pytest.mark.parametrize(
    "query",
    [
        "",
        "()",
        "a AND",
        "a OR",
        "NOT",
        "a)",
        "(a",
        '"a',
        '""',
        "OR a",
        "a AND OR b",
        "a NOT OR b",
        "!!!",
        "well-known",
        "x" * 4097,
        "(" * 200 + "a" + ")" * 200,
    ],
)
def test_parser_errors(query):
    with pytest.raises(ValueError):
        parse(query)


@pytest.mark.parametrize("storage", ["plain", "slots", "array"])
def test_phrase_order_repeats_and_negation(storage):
    index = build(["a b a a b", "b a", "a x b", ""], storage=storage)
    assert parse('"a b"').evaluate(index) == {0}
    assert parse('"a a b"').evaluate(index) == {0}
    assert parse('"b a"').evaluate(index) == {0, 1}
    assert parse('a NOT "a b"').evaluate(index) == {1, 2}
    assert parse("NOT missing").evaluate(index) == {0, 1, 2, 3}
    assert index.positions("a", 0) == (0, 2, 3)
    with pytest.raises(ValueError, match="positions"):
        parse('"a b"').evaluate(build(["a b"], positions=False))


@pytest.mark.parametrize("format", ["pickle", "json"])
@pytest.mark.parametrize("storage", ["plain", "slots", "array"])
def test_position_roundtrip_and_cleanup(tmp_path, format, storage):
    original = build(["a a b", ""], storage=storage)
    original.matched_ids('"a b"')
    path = tmp_path / f"index.{format}"
    save(original, path, format)
    with pytest.raises(RuntimeError, match="навмисно"):
        with open_index(path, format) as index:
            assert index == original
            assert index.cache_info().currsize == 0
            assert index.matched_ids('"a b"') == frozenset({0})
            assert index.texts[0] == "a a b"
            raise RuntimeError("навмисно")
    assert index.closed and index.cache_info().currsize == 0
    assert not index._texts and not index._postings
    with pytest.raises(ValueError, match="закрито"):
        len(index)
    index.close()


def test_legacy_json(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        '{"version":1,"storage":"slots","documents":[[0,1,"a","a"]],'
        '"postings":{"word":[[0,1]]}}'
    )
    index = load(path)
    assert not index.has_positions
    assert search(index, "word")[0].doc_id == 0


def test_bm25_formula_and_length():
    index = build(["rare", "common " * 100, "common"])
    scorer = BM25()
    expected = (
        math.log1p((3 - 1 + 0.5) / (1 + 0.5)) * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 1 / 34))
    )
    assert scorer.score("rare", Posting(0, 1), index) == pytest.approx(expected)
    short = scorer.score("common", Posting(2, 1), index)
    long = scorer.score("common", Posting(1, 1), index)
    assert short > long
    assert BM25(b=0).score("common", Posting(2, 1), index) == BM25(b=0).score(
        "common", Posting(1, 1), index
    )
    gains = [BM25(b=0).score("rare", Posting(0, tf), index) for tf in (1, 2, 19, 20)]
    assert (gains[3] - gains[2]) < 0.02 * (gains[1] - gains[0])
    assert TfIdf().score("rare", Posting(0, 1), index) == pytest.approx(math.log(4))


@pytest.mark.parametrize(
    "kwargs",
    [{"b": -1}, {"b": 2}, {"b": float("nan")}, {"k1": 0}, {"k1": float("inf")}],
)
def test_bm25_parameters(kwargs):
    with pytest.raises(ValueError):
        BM25(**kwargs)


def test_ranked_results_protocol_and_topk():
    index = build(["common", "common rare", "common", ""])
    assert search(index, "rare OR common", BM25(), 1)[0].doc_id == 1
    assert search(index, "rare OR common", TfIdf(), 1)[0].doc_id == 1

    class Custom:
        def score(self, term, posting, index):
            return float(posting.doc_id)

    assert [r.doc_id for r in search(index, "common", Custom(), 2)] == [2, 1]
    assert search(index, "common", k=0) == []
    assert search(index, "missing") == []
    assert search(build([]), "NOT missing") == []
    assert [r.doc_id for r in search(index, "NOT missing")] == [0, 1, 2, 3]
    assert search(index, "NOT common")[0].score == 0
    with pytest.raises(ValueError):
        search(index, "common", k=-1)
    a, b = SearchResult(1, 1.0, "a"), SearchResult(0, 2.0, "b")
    assert sorted([b, a]) == [a, b]
    assert "2.00000" in str(b)


@pytest.mark.parametrize("scorer", [BM25(b=0), TfIdf()])
def test_rare_term_and_diminishing_gain(scorer):
    index = build(["rare common", "common", "common"])
    assert scorer.score("rare", Posting(0, 1), index) > scorer.score(
        "common", Posting(0, 1), index
    )
    values = [scorer.score("rare", Posting(0, tf), index) for tf in (1, 2, 19, 20)]
    assert 0 < values[3] - values[2] < 0.1 * (values[1] - values[0])


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
def test_length_normalization_difference(scorer):
    index = build(["word", "word " + "other " * 1000])
    short = scorer.score("word", Posting(0, 1), index)
    long = scorer.score("word", Posting(1, 1), index)
    if isinstance(scorer, BM25):
        assert short > long
    else:
        # Ця формула TF-IDF не нормалізує довжину: очікувана нічия.
        assert short == long


def test_cache_isolation_and_timing(caplog):
    with caplog.at_level(logging.DEBUG, logger="findex"):
        first = build(["a"])
        second = build(["b"])
        search(first, "a")
        search(first, "a", TfIdf())
        assert search(second, "a") == []
    assert first.cache_info().hits == 1 and second.cache_info().hits == 0
    assert "cache hit" in caplog.text and "cache miss" in caplog.text
    assert "build_index:" in caplog.text and "search:" in caplog.text
    assert search.__name__ == "search" and hasattr(search, "__wrapped__")

    @timed
    def broken():
        """Опис збережено."""
        raise RuntimeError("перевірка")

    with caplog.at_level(logging.DEBUG, logger="findex"), pytest.raises(RuntimeError):
        broken()
    assert "broken:" in caplog.text and broken.__doc__ == "Опис збережено."


def test_snippets_unicode_boundaries_and_best_window():
    text = "cat " + "x " * 100 + "cat dog Straße cafe\u0301 п’ять concatenate"
    result = snippet(text, {"cat", "dog", "strasse", "café", "п'ять"})
    assert "[cat] [dog] [Straße] [cafe\u0301] [п’ять]" in result
    assert "con[cat]" not in result and result.startswith("…")
    assert snippet("begin end", {"begin"}) == "[begin] end"
    assert snippet("begin end", {"end"}) == "begin [end]"
    assert snippet("hello", set()) == "hello"
    assert snippet("a " * 100, set()).endswith("…")


def test_positions_validation(tmp_path):
    index = build(["a b"])
    index._positions["a"][0] = (99,)
    path = tmp_path / "invalid.bin"
    save(index, path)
    with pytest.raises(ValueError, match="позиції"):
        load(path)


def test_cache_limit_and_fresh_results():
    index = build(["a"])
    for i in range(257):
        index.matched_ids(f"unknown{i}")
    assert index.cache_info().currsize == 256
    results = search(index, "a")
    results.clear()
    assert len(search(index, "a")) == 1


def test_ranked_cli(tmp_path, capsys, caplog):
    from findex.search import main

    path = tmp_path / "index.bin"
    save(build(["python async event loop", "python java event loop"]), path)
    with caplog.at_level(logging.DEBUG, logger="findex"):
        main(
            [
                str(path),
                'python (async OR await) NOT java "event loop"',
                "--top",
                "1",
                "--repeat",
                "2",
                "--verbose",
            ]
        )
    output = capsys.readouterr().out
    assert "[event] [loop]" in output
    assert "Знайдено документів: 1" in output
    assert "cache hit" in caplog.text and "load:" in caplog.text
