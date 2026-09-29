"""Тестовий індекс лише в пам'яті; реального файла немає."""

import logging
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli
from findex.corpus import Document
from findex.index import build_index
from findex.web import create_app, get_index
from findex.web.settings import Settings, get_settings


@pytest.fixture
def web_index():
    index = build_index(
        [
            Document("first", Path("first.txt"), "Python event loop " * 4, "Перша"),
            Document("second", Path("second.txt"), "Python event loop", "Друга"),
            Document(
                "unsafe",
                Path("unsafe.txt"),
                "<script>alert(1)</script> python [<img src=x onerror=alert(1)>]",
                "<svg onload=alert(1)>",
            ),
        ],
        positions=True,
    )
    yield index
    index.close()


@pytest.fixture
def application(web_index):
    app = create_app()
    app.dependency_overrides[get_index] = lambda: web_index
    app.dependency_overrides[get_settings] = lambda: Settings(index_path=Path("unused"))
    return app


@pytest.fixture
def client(application):
    with TestClient(application) as client:
        yield client


def test_pagination(client):
    first = client.get("/search", params={"q": "python", "k": 1}).json()
    second = client.get("/search", params={"q": "python", "k": 1, "page": 2}).json()
    assert first["total"] == first["pages"] == 3
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]
    assert first["took_ms"] >= 0
    empty = client.get("/search", params={"q": "python", "page": 10}).json()
    assert empty["results"] == [] and empty["total"] == 3


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "   "},
        {"q": "x" * 301},
        {"q": "python", "k": 0},
        {"q": "python", "k": 101},
        {"q": "python", "k": "abc"},
        {"q": "python", "scorer": "bad"},
        {"q": "python", "page": 0},
        {"q": "python", "page": 10001},
        {"q": "("},
        {"q": "python AND"},
    ],
)
def test_validation(client, params):
    response = client.get("/search", params=params)
    assert response.status_code == 422 and "detail" in response.json()


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_scorers(client, scorer):
    response = client.get("/search", params={"q": '"event loop"', "scorer": scorer})
    assert response.status_code == 200 and response.json()["total"] == 2


def test_docs_stats_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/stats").json()["documents"] == 3
    assert client.get("/docs/0").json()["title"] == "Перша"
    assert client.get("/docs/999").status_code == 404
    assert client.get("/doc/999").status_code == 404
    assert client.get("/docs/abc").status_code == 422
    assert client.get("/docs").status_code == 200
    assert "/search" in client.get("/openapi.json").json()["paths"]


def test_not_loaded():
    # Без входу в lifespan: файл індексу не відкривається.
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 503


def test_missing_index_dependency(application):
    def unavailable():
        raise HTTPException(503, "Не готово")

    application.dependency_overrides[get_index] = unavailable
    with TestClient(application) as client:
        assert client.get("/health").status_code == 503


def test_unexpected_exception(application, caplog):
    def broken():
        raise RuntimeError("private-secret-path")

    application.dependency_overrides[get_index] = broken
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        with TestClient(application) as client:
            response = client.get("/stats")
    assert response.status_code == 500
    assert (
        "private-secret-path" not in response.text and "Traceback" not in response.text
    )
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["X-Request-ID"] in caplog.text
    assert "private-secret-path" in caplog.text


def test_html_escape_and_navigation(client):
    response = client.get("/", params={"q": "python", "k": 1})
    assert response.status_code == 200
    assert "<mark>" in response.text and "page=2" in response.text
    doc = client.get("/doc/2")
    assert "<script>" not in doc.text and "&lt;script&gt;" in doc.text
    assert "<svg onload" not in doc.text
    home = client.get("/", params={"q": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in home.text
    assert client.get("/static/style.css").status_code == 200
    assert "Пошук" in client.get("/").text or "пошук" in client.get("/").text
    assert "Перевірте" in client.get("/", params={"q": "x" * 301}).text


def test_lifespan_once_and_cleanup(monkeypatch):
    import findex.web as web

    index = build_index([Document("x", Path("x.txt"), "python")])
    calls = []
    monkeypatch.setattr(web, "load", lambda path: calls.append(path) or index)
    # Лише stat підмінено, тест не створює файл індексу.
    from types import SimpleNamespace

    original = Path.stat
    monkeypatch.setattr(
        Path,
        "stat",
        lambda self, **kwargs: (
            SimpleNamespace(st_size=123)
            if str(self) == "virtual-index"
            else original(self, **kwargs)
        ),
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        index_path=Path("virtual-index")
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/stats").json()["index_bytes"] == 123
        assert len(calls) == 1
    assert index.closed


def test_settings_and_serve(monkeypatch):
    import uvicorn

    monkeypatch.setenv("INDEX_PATH", "from-env.json")
    monkeypatch.setenv("PORT", "8765")
    get_settings.cache_clear()
    calls = []
    monkeypatch.setattr(
        uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    assert Settings().index_path == Path("from-env.json")
    result = CliRunner().invoke(cli, ["serve", "--workers", "4"])
    assert result.exit_code == 0, result.output
    assert calls[0][1]["workers"] == 4 and calls[0][1]["port"] == 8765
    get_settings.cache_clear()
