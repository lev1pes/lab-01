"""Обмежена черга адрес і потік сторінок з керованим завершенням."""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

import anyio
import httpx

from findex.crawler.fetch import CrawlStats, RequestGate, fetch, logger
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, parse_html


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


def prepare_client(
    transport: httpx.AsyncBaseTransport | None, concurrency: int
) -> httpx.AsyncClient:
    """TLS-файли та перше завантаження backend не займають потік event loop."""

    async def warmup() -> None:
        await anyio.sleep(0)

    anyio.run(warmup)
    return httpx.AsyncClient(
        transport=transport,
        timeout=None,
        trust_env=False,
        limits=httpx.Limits(max_connections=concurrency),
    )


async def crawl(
    seeds: Sequence[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    delay: float = 0.2,
    allowed_domains: Sequence[str] | None = None,
    timeout: float = 10.0,
    stats: CrawlStats | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncGenerator[Page, None]:
    if max_pages < 0 or timeout <= 0:
        raise ValueError("Некоректний бюджет сторінок або таймаут")
    state = stats if stats is not None else CrawlStats()
    gate = RequestGate(concurrency, per_host, delay, state)
    canonical_seeds = [canonicalize(url) for url in seeds]
    domains = (
        set(allowed_domains)
        if allowed_domains is not None
        else {urlsplit(url).hostname or "" for url in canonical_seeds}
    )
    domains = {domain.lower().encode("idna").decode("ascii") for domain in domains}
    frontier: asyncio.Queue[str] = asyncio.Queue()
    output: asyncio.Queue[Page | Exception | None] = asyncio.Queue(concurrency)
    seen: set[str] = set()
    # Циклічний сайт із безліччю query не може безмежно ростити frontier.
    max_urls = max_pages * 20
    stopped = asyncio.Event()
    reserved = 0

    def enqueue(raw: str, base: str = "") -> None:
        try:
            url = canonicalize(raw, base)
        except ValueError:
            return
        if (
            urlsplit(url).hostname not in domains
            or url in seen
            or len(seen) >= max_urls
        ):
            return
        seen.add(url)
        frontier.put_nowait(url)
        state.queued = frontier.qsize()

    if max_pages == 0:
        return
    for seed in canonical_seeds:
        enqueue(seed)

    preparing = asyncio.create_task(
        asyncio.to_thread(prepare_client, transport, concurrency)
    )
    try:
        client = await asyncio.shield(preparing)
    except asyncio.CancelledError:
        client = await preparing
        await client.aclose()
        raise
    async with client:
        robots = RobotsCache(client, gate, timeout)

        async def worker() -> None:
            nonlocal reserved
            while not stopped.is_set():
                url = await frontier.get()
                state.queued = frontier.qsize()
                try:
                    if not await robots.allowed(url):
                        state.blocked += 1
                        logger.info(
                            "url=%s status=robots-denied bytes=0 elapsed=0", url
                        )
                        continue
                    result = await fetch(client, url, gate=gate, timeout=timeout)
                    if result.error:
                        continue
                    if result.status in (301, 302, 303, 307, 308):
                        # Не авто-follow: нова адреса знову проходить allowlist/robots.
                        enqueue(result.headers.get("location", ""), url)
                        continue
                    content_type = result.headers.get("content-type", "").lower()
                    if result.status != 200 or not (
                        "text/html" in content_type
                        or "application/xhtml+xml" in content_type
                    ):
                        continue

                    def parse() -> tuple[str, str, list[str]]:
                        text = httpx.Response(
                            200, headers=result.headers, content=result.body
                        ).text
                        return parse_html(text)

                    title, text, links = await asyncio.to_thread(parse)
                    if reserved >= max_pages:
                        continue
                    reserved += 1
                    for link in links:
                        enqueue(link, url)
                    await output.put(
                        Page(url, title, text, datetime.now(UTC).isoformat())
                    )
                    state.pages += 1
                    if state.pages >= max_pages:
                        stopped.set()
                finally:
                    frontier.task_done()

        async def produce() -> None:
            try:
                async with asyncio.TaskGroup() as group:
                    workers = [group.create_task(worker()) for _ in range(concurrency)]
                    joined = group.create_task(frontier.join())
                    capped = group.create_task(stopped.wait())
                    try:
                        await asyncio.wait(
                            (joined, capped), return_when=asyncio.FIRST_COMPLETED
                        )
                    finally:
                        for task in [*workers, joined, capped]:
                            task.cancel()
            except Exception as error:
                await output.put(error)
            else:
                await output.put(None)

        producer = asyncio.create_task(produce(), name="findex-crawl")
        try:
            while True:
                item = await output.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
            while not frontier.empty():
                frontier.get_nowait()
                frontier.task_done()
            state.queued = 0
