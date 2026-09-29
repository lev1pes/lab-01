"""Всі HTTP-відповіді підмінені MockTransport; живої мережі немає."""

import asyncio
import importlib
import io
import json
from collections import Counter
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from time import perf_counter

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.crawler.crawl import crawl
from findex.crawler.fetch import (
    USER_AGENT,
    CrawlStats,
    RequestGate,
    fetch,
    retry_after,
)
from findex.crawler.output import collect, crawl_log
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, parse_html
from findex.store import load


def run(coroutine):
    return asyncio.run(coroutine, debug=True)


def html(text):
    return httpx.Response(200, text=text, headers={"content-type": "text/html"})


def robots():
    return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")


@pytest.mark.parametrize(
    "source,expected",
    [
        ("HTTP://Example.COM:80/a/?z=2&a=1#x", "http://example.com/a?a=1&z=2"),
        ("https://EXAMPLE.COM:443/", "https://example.com/"),
        ("https://example.com:8443/a///", "https://example.com:8443/a"),
        ("http://[::1]:80/x", "http://[::1]/x"),
        ("https://example.com/?a=&a=1", "https://example.com/?a=&a=1"),
    ],
)
def test_canonical(source, expected):
    assert canonicalize(source) == expected


@pytest.mark.parametrize(
    "url",
    [
        "mailto:a@b",
        "javascript:void(0)",
        "file:///x",
        "http:///",
        "http://a:bad",
        "https://u:p@a",
    ],
)
def test_invalid_url(url):
    with pytest.raises(ValueError):
        canonicalize(url)


def test_html():
    title, text, links = parse_html(
        "<title>Кіт &amp; Python</title><p>текст</p><script>secret</script>"
        '<style>hide</style><a href="/x">X</a>'
    )
    assert title == "Кіт & Python"
    assert "secret" not in text and "hide" not in text and "текст" in text
    assert links == ["/x"]
    assert canonicalize("../x#y", "https://a/dir/file") == "https://a/x"


@pytest.mark.parametrize(
    "status,attempts",
    [(200, 1), (404, 1), (403, 1), (400, 1), (429, 3), (500, 3), (503, 3)],
)
def test_retry_status(status, attempts):
    async def check():
        count = 0

        async def handler(request):
            nonlocal count
            count += 1
            assert request.headers["user-agent"] == USER_AGENT
            return httpx.Response(status, content=b"abc")

        state = CrawlStats()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client, "https://a/", gate=RequestGate(2, 1, 0, state), base_delay=0
            )
        assert count == attempts and result.status == status and result.body == b"abc"
        assert state.in_flight == 0 and state.requests == attempts

    run(check())


def test_retry_after():
    assert retry_after("2") == 2
    assert retry_after("-2") == 0
    assert retry_after(None) == retry_after("bad") == 0
    assert (
        1 < retry_after(format_datetime(datetime.now(UTC) + timedelta(seconds=3))) <= 3
    )


def test_retry_after_applied(monkeypatch):
    module = importlib.import_module("findex.crawler.fetch")
    monkeypatch.setattr(module, "retry_after", lambda value: 0.025)

    async def check():
        times = []

        async def handler(request):
            times.append(perf_counter())
            return httpx.Response(
                429 if len(times) == 1 else 200, headers={"Retry-After": "1"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://a/",
                gate=RequestGate(1, 1, 0, CrawlStats()),
                base_delay=0,
            )
        assert result.status == 200 and times[1] - times[0] >= 0.02

    run(check())


@pytest.mark.parametrize("mode,expected", [("timeout", 3), ("connect", 1), ("big", 1)])
def test_fetch_failures(mode, expected):
    async def check():
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if mode == "timeout":
                await asyncio.sleep(1)
            if mode == "connect":
                raise httpx.ConnectError("broken", request=request)
            return httpx.Response(200, content=b"too large")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://a/",
                timeout=0.002,
                max_bytes=2,
                base_delay=0,
                gate=RequestGate(1, 1, 0, CrawlStats()),
            )
        assert result.error and calls == expected

    run(check())


@pytest.mark.parametrize(
    "status,allowed",
    [(200, True), (404, True), (403, False), (401, False), (302, False)],
)
def test_robots_cache(status, allowed):
    async def check():
        calls = Counter()

        async def handler(request):
            calls[str(request.url)] += 1
            return httpx.Response(
                status, text="User-agent: *\nDisallow: /private\nCrawl-delay: 1\n"
            )

        gate = RequestGate(10, 2, 0, CrawlStats())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cache = RobotsCache(client, gate)
            values = await asyncio.gather(
                *(cache.allowed("https://a/page") for _ in range(5))
            )
            assert values == [allowed] * 5
            assert await cache.allowed("https://b/page") == allowed
            if status == 200:
                assert not await cache.allowed("https://a/private/x")
                assert gate.host("https://a/").delay == 1
        assert calls == {"https://a/robots.txt": 1, "https://b/robots.txt": 1}

    run(check())


def test_crawl_graph_and_redirects():
    async def check():
        calls = Counter()

        async def handler(request):
            url = str(request.url)
            calls[url] += 1
            if request.url.path == "/robots.txt":
                return robots()
            if request.url.path == "/":
                return html(
                    '<a href="/a/#one">a</a><a href="/a#two">dup</a>'
                    '<a href="/private">no</a><a href="https://outside/b">out</a>'
                    '<a href="/r">redirect</a><a href="/loop">loop</a>'
                    '<a href="/image">img</a>'
                )
            if request.url.path == "/r":
                return httpx.Response(
                    302, headers={"location": "https://outside/secret"}
                )
            if request.url.path == "/loop":
                return httpx.Response(302, headers={"location": "/loop"})
            if request.url.path == "/image":
                return httpx.Response(200, headers={"content-type": "image/png"})
            return html('<title>Python</title><a href="/">home</a>event loop')

        state = CrawlStats()
        pages = [
            p
            async for p in crawl(
                ["https://a/"],
                max_pages=20,
                concurrency=5,
                delay=0,
                stats=state,
                transport=httpx.MockTransport(handler),
            )
        ]
        assert {p.url for p in pages} == {"https://a/", "https://a/a"}
        assert all(n == 1 for n in calls.values())
        assert not any("outside" in u or "/private" in u for u in calls)
        assert state.blocked == 1 and state.in_flight == state.queued == 0

    run(check())


@pytest.mark.parametrize("early", [False, True])
def test_cap_and_early_close(early):
    async def check():
        class Transport(httpx.MockTransport):
            closed = False

            async def aclose(self):
                self.closed = True

        async def handler(request):
            if request.url.path == "/robots.txt":
                return robots()
            await asyncio.sleep(0.002)
            return html("".join(f'<a href="/{i}">link</a>' for i in range(50)))

        transport = Transport(handler)
        state = CrawlStats()
        pages = []
        async with aclosing(
            crawl(
                ["https://a/"],
                max_pages=3,
                concurrency=10,
                delay=0,
                stats=state,
                transport=transport,
            )
        ) as stream:
            async for page in stream:
                pages.append(page)
                if early:
                    break
        assert len(pages) == (1 if early else 3)
        assert transport.closed and state.in_flight == state.queued == 0
        assert len(asyncio.all_tasks()) == 1

    run(check())


def test_taskgroup_failure_cancels_siblings(monkeypatch):
    module = importlib.import_module("findex.crawler.crawl")

    def broken(text):
        raise RuntimeError("parser broke")

    monkeypatch.setattr(module, "parse_html", broken)

    async def check():
        transport = httpx.MockTransport(
            lambda req: robots() if req.url.path == "/robots.txt" else html("x")
        )
        with pytest.raises(ExceptionGroup, match="TaskGroup"):
            async with asyncio.timeout(1):
                _ = [
                    p async for p in crawl(["https://a/"], transport=transport, delay=0)
                ]
        assert len(asyncio.all_tasks()) == 1

    run(check())


def test_global_host_limits_and_delay():
    async def check():
        active = Counter()
        peaks = Counter()
        starts = {"a": [], "b": []}
        total_peak = 0

        async def handler(request):
            nonlocal total_peak
            host = request.url.host
            starts[host].append(perf_counter())
            active[host] += 1
            peaks[host] = max(peaks[host], active[host])
            total_peak = max(total_peak, sum(active.values()))
            await asyncio.sleep(0.015)
            active[host] -= 1
            return httpx.Response(200)

        gate = RequestGate(3, 2, 0.003, CrawlStats())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await asyncio.gather(
                *(
                    fetch(client, f"https://{host}/{i}", gate=gate)
                    for i in range(6)
                    for host in ("a", "b")
                )
            )
        assert total_peak <= 3 and max(peaks.values()) <= 2
        assert total_peak > 1
        for times in starts.values():
            assert all(b - a >= 0.002 for a, b in zip(times, times[1:]))

    run(check())


def test_jsonl_stream_and_cli(tmp_path, monkeypatch):
    async def handler(request):
        return (
            robots()
            if request.url.path == "/robots.txt"
            else html("<title>Асинхронність</title>Python event loop")
        )

    module = importlib.import_module("findex.cli")
    original = collect

    async def mocked(*args, **kwargs):
        await original(*args, **kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(module, "collect", mocked)
    path, log_path = tmp_path / "crawl.jsonl", tmp_path / "crawl.log"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "crawl",
            "https://a/",
            "--out",
            str(path),
            "--log",
            str(log_path),
            "--max-pages",
            "1",
            "--delay",
            "0",
            "--debug",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Черга" in result.stderr and "Помилок" in result.stderr
    assert "status=200" in log_path.read_text(encoding="utf-8")
    doc = next(iter_documents(path))
    assert doc.doc_id == "https://a/" and doc.title == "Асинхронність"
    index_path = tmp_path / "index.json"
    result = runner.invoke(
        app, ["index", str(path), "--out", str(index_path), "--positions"]
    )
    assert result.exit_code == 0, result.output
    assert load(index_path).doc_meta[0].title == "Асинхронність"
    result = runner.invoke(app, ["search", str(index_path), '"event loop"', "--json"])
    assert json.loads(result.stdout)["path"] == "https://a/"
    assert (
        runner.invoke(
            app, ["index", str(path), "--out", str(index_path), "--executor", "threads"]
        ).exit_code
        == 1
    )


def test_jsonl_lazy_validation(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(
        '\n{"url":"https://a/", "title":"A", "text":"one"}\ninvalid\n', encoding="utf-8"
    )
    stream = iter_documents(path)
    assert next(stream).text == "one"
    with pytest.raises(ValueError, match="рядок 3"):
        next(stream)


@pytest.mark.parametrize(
    "data", ["[]", '{"url":1}', '{"url":"x","title":"a","text":3}']
)
def test_jsonl_invalid(tmp_path, data):
    path = tmp_path / "x.jsonl"
    path.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_documents(path))


def test_empty_and_invalid_config():
    async def check():
        assert [p async for p in crawl([], max_pages=0)] == []
        assert [p async for p in crawl([])] == []
        for options in (
            {"max_pages": -1},
            {"concurrency": 0},
            {"per_host": 0},
            {"delay": -1},
            {"timeout": 0},
        ):
            with pytest.raises(ValueError):
                _ = [p async for p in crawl([], **options)]

    run(check())


def test_stream_writer(tmp_path):
    async def check():
        target = io.StringIO()
        state = CrawlStats()
        transport = httpx.MockTransport(
            lambda req: robots() if req.url.path == "/robots.txt" else html("Python")
        )
        with crawl_log(tmp_path / "crawl.log"):
            await collect(
                "https://a/",
                target,
                stats=state,
                max_pages=1,
                concurrency=2,
                per_host=1,
                delay=0,
                timeout=1,
                transport=transport,
            )
        assert len(target.getvalue().splitlines()) == 1

    run(check())


def test_fetch_backoff_and_jitter(monkeypatch):
    module = importlib.import_module("findex.crawler.fetch")
    jitter_calls = []

    def jitter(low, high):
        jitter_calls.append((low, high))
        return high / 2

    monkeypatch.setattr(module.random, "uniform", jitter)

    async def check():
        starts = []

        def handler(request):
            starts.append(perf_counter())
            return httpx.Response(503 if len(starts) < 3 else 200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await fetch(
                client,
                "https://a/",
                gate=RequestGate(1, 1, 0, CrawlStats()),
                base_delay=0.01,
            )
        assert jitter_calls == [(0, 0.01), (0, 0.01)]
        assert starts[1] - starts[0] >= 0.015
        assert starts[2] - starts[1] >= 0.025

    run(check())


def test_external_cancel_during_request():
    async def check():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def handler(request):
            if request.url.path == "/robots.txt":
                return robots()
            entered.set()
            try:
                await asyncio.sleep(100)
            finally:
                cancelled.set()
            return html("never")

        state = CrawlStats()

        async def consume():
            async with aclosing(
                crawl(
                    ["https://a/"],
                    delay=0,
                    stats=state,
                    transport=httpx.MockTransport(handler),
                )
            ) as stream:
                return [page async for page in stream]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set() and state.in_flight == state.queued == 0
        assert len(asyncio.all_tasks()) == 1

    run(check())


def test_failed_page_does_not_stop_other_pages():
    async def check():
        def handler(request):
            if request.url.path == "/robots.txt":
                return robots()
            if request.url.path == "/bad":
                raise httpx.ConnectError("broken", request=request)
            return html('<a href="/bad">broken</a><a href="mailto:a@b">skip</a>')

        state = CrawlStats()
        pages = [
            p
            async for p in crawl(
                ["https://a/"],
                delay=0,
                stats=state,
                transport=httpx.MockTransport(handler),
            )
        ]
        assert len(pages) == 1 and state.errors == 1

    run(check())


def test_fetch_invalid_parameters():
    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: html("x"))
        ) as client:
            for options in ({"timeout": 0}, {"attempts": 4}, {"max_bytes": 0}):
                with pytest.raises(ValueError):
                    await fetch(
                        client,
                        "https://a/",
                        gate=RequestGate(1, 1, 0, CrawlStats()),
                        **options,
                    )

    run(check())


def test_retry_after_holds_already_waiting_request(monkeypatch):
    module = importlib.import_module("findex.crawler.fetch")
    monkeypatch.setattr(module, "retry_after", lambda value: 0.03)

    async def check():
        starts = []

        async def handler(request):
            starts.append(perf_counter())
            await asyncio.sleep(0.002)
            return httpx.Response(429 if len(starts) == 1 else 200)

        gate = RequestGate(1, 2, 0, CrawlStats())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await asyncio.gather(
                fetch(client, "https://a/first", gate=gate, base_delay=0),
                fetch(client, "https://a/second", gate=gate, base_delay=0),
            )
        assert starts[1] - starts[0] >= 0.03
        assert gate.stats.in_flight == 0

    run(check())
