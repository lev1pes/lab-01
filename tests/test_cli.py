import json
import logging

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.store import load

runner = CliRunner()


@pytest.fixture(autouse=True)
def restore_logging():
    log = logging.getLogger("findex")
    handlers, level, propagate = log.handlers[:], log.level, log.propagate
    yield
    for handler in log.handlers:
        if handler not in handlers:
            handler.close()
    log.handlers = handlers
    log.setLevel(level)
    log.propagate = propagate


@pytest.mark.slow
@pytest.mark.parametrize("command", [[], ["index"], ["search"], ["stats"]])
def test_help(command):
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.stdout


@pytest.mark.parametrize("suffix", ["bin", "json"])
@pytest.mark.parametrize("limit", [0, 2, None])
def test_index(tiny_corpus, tmp_path, suffix, limit):
    path = tmp_path / f"output.{suffix}"
    args = ["index", str(tiny_corpus), "--out", str(path), "--positions"]
    if limit is not None:
        args += ["--limit", str(limit)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Індексація" in result.stderr
    assert "Індексація" not in result.stdout
    index = load(path)
    assert index.num_docs == (4 if limit is None else limit)
    assert index.has_positions


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_search_table(index_path, scorer):
    result = runner.invoke(
        app,
        [
            "search",
            str(index_path),
            'python (async OR await) NOT java "event loop"',
            "--k",
            "5",
            "--scorer",
            scorer,
        ],
        env={"COLUMNS": "160"},
    )
    assert result.exit_code == 0, result.output
    assert "a.txt" in result.stdout and "b.txt" not in result.stdout
    assert "[event] [loop]" in result.stdout
    assert "Бал" in result.stdout


@pytest.mark.parametrize(
    "verbose, info, debug",
    [(None, False, False), ("-v", True, False), ("-vv", True, True)],
)
def test_json_and_logging(index_path, verbose, info, debug):
    args = ([verbose] if verbose else []) + [
        "search",
        str(index_path),
        "кіт",
        "--json",
        "--k",
        "1",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 1
    assert rows[0]["path"] == "c.txt" and rows[0]["score"] > 0
    assert "[кіт]" in rows[0]["snippet"]
    assert ("Час:" in result.stderr) == info
    assert ("search:" in result.stderr) == debug
    assert "INFO" not in result.stdout


@pytest.mark.parametrize("query,k", [("missing", 5), ("python", 0)])
def test_empty_json(index_path, query, k):
    result = runner.invoke(
        app, ["search", str(index_path), query, "--json", "--k", str(k)]
    )
    assert result.exit_code == 0
    assert result.stdout == ""


def test_stats(index_path):
    result = runner.invoke(app, ["stats", str(index_path)])
    assert result.exit_code == 0, result.output
    assert "Документів" in result.stdout and "Топ-50" in result.stdout
    assert "python" in result.stdout


@pytest.mark.parametrize("query", ["(", "python AND", '"event', ""])
def test_invalid_query_one_line(index_path, query):
    result = runner.invoke(app, ["search", str(index_path), query, "--json"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert len(result.stderr.strip().splitlines()) == 1
    assert "Помилка" in result.stderr and "Traceback" not in result.output


@pytest.mark.parametrize("command", ["search", "stats", "index"])
def test_missing_file(tmp_path, command):
    args = [command, str(tmp_path / "missing")]
    args += (
        ["word"]
        if command == "search"
        else ["--out", str(tmp_path / "out.bin")]
        if command == "index"
        else []
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "не знайдено" in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


@pytest.mark.parametrize("option,value", [("--k", "-1"), ("--scorer", "other")])
def test_invalid_options(index_path, option, value):
    result = runner.invoke(app, ["search", str(index_path), "python", option, value])
    assert result.exit_code != 0 and "Traceback" not in result.output


def test_corrupt_index(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"broken")
    result = runner.invoke(app, ["stats", str(path)])
    assert result.exit_code == 1 and "Пошкоджений" in result.stderr


def test_phrase_requires_positions(tiny_corpus, tmp_path):
    path = tmp_path / "plain.bin"
    assert (
        runner.invoke(app, ["index", str(tiny_corpus), "--out", str(path)]).exit_code
        == 0
    )
    result = runner.invoke(app, ["search", str(path), '"event loop"'])
    assert result.exit_code == 1 and "--positions" in result.stderr
