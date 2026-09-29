"""Спільні ліміти застосовуються до кожної спроби, включно з robots.txt."""

import asyncio
import logging
import random
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import perf_counter

import httpx

from findex.crawler.urls import origin

USER_AGENT = "findex/0.6 (+https://github.com/lev1pes/lab-01/issues)"
logger = logging.getLogger("findex.crawl")


@dataclass(slots=True)
class CrawlStats:
    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queued: int = 0
    blocked: int = 0
    requests: int = 0
    started: float = field(default_factory=perf_counter)

    @property
    def rate(self) -> float:
        return self.pages / max(0.001, perf_counter() - self.started)


@dataclass(slots=True)
class HostLimit:
    semaphore: asyncio.Semaphore
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_start: float = 0.0
    delay: float = 0.0


class RequestGate:
    def __init__(
        self, concurrency: int, per_host: int, delay: float, stats: CrawlStats
    ) -> None:
        if concurrency < 1 or per_host < 1 or delay < 0:
            raise ValueError("Некоректні ліміти запитів")
        self.global_sem = asyncio.Semaphore(concurrency)
        self.per_host = per_host
        self.delay = delay
        self.stats = stats
        self.hosts: dict[str, HostLimit] = {}

    def host(self, url: str) -> HostLimit:
        key = origin(url)
        if key not in self.hosts:
            self.hosts[key] = HostLimit(
                asyncio.Semaphore(self.per_host), delay=self.delay
            )
        return self.hosts[key]

    @asynccontextmanager
    async def slot(self, url: str) -> AsyncGenerator[None, None]:
        host = self.host(url)
        async with host.semaphore:
            # Lock захищає перевірку часу й початок наступного запиту.
            async with host.lock:
                while host.next_start > perf_counter():
                    await asyncio.sleep(host.next_start - perf_counter())
                await self.global_sem.acquire()
                try:
                    # Поки чекали глобальний слот, інший запит міг отримати 429.
                    while host.next_start > perf_counter():
                        await asyncio.sleep(host.next_start - perf_counter())
                except BaseException:
                    self.global_sem.release()
                    raise
                host.next_start = perf_counter() + host.delay
            self.stats.in_flight += 1
            self.stats.requests += 1
            try:
                yield
            finally:
                self.stats.in_flight -= 1
                self.global_sem.release()


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int | None
    body: bytes
    headers: httpx.Headers
    elapsed: float
    error: str = ""


def retry_after(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return max(0.0, float(int(value)))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            return max(0.0, (date - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.0


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    gate: RequestGate,
    timeout: float = 10.0,
    attempts: int = 3,
    base_delay: float = 0.25,
    max_bytes: int = 2 * 1024 * 1024,
) -> FetchResult:
    """До трьох спроб загалом; мережеві помилки без таймауту не повторюються."""
    if timeout <= 0 or not 1 <= attempts <= 3 or max_bytes < 1 or base_delay < 0:
        raise ValueError("Некоректні параметри fetch")
    for attempt in range(attempts):
        body = bytearray()
        status: int | None = None
        headers = httpx.Headers()
        error = ""
        retry = False
        async with gate.slot(url):
            started = perf_counter()
            try:
                async with asyncio.timeout(timeout):
                    async with client.stream(
                        "GET",
                        url,
                        follow_redirects=False,
                        headers={"User-Agent": USER_AGENT},
                    ) as response:
                        status, headers = response.status_code, response.headers
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > max_bytes:
                                error = "body-too-large"
                                break
                            body.extend(chunk)
                retry = not error and (status == 429 or 500 <= (status or 0) < 600)
            except (TimeoutError, httpx.TimeoutException):
                error, retry = "timeout", True
            except httpx.HTTPError as exc:
                error = type(exc).__name__
            except asyncio.CancelledError:
                logger.info(
                    "url=%s status=cancelled bytes=%d elapsed=%.4f attempt=%d",
                    url,
                    len(body),
                    perf_counter() - started,
                    attempt + 1,
                )
                raise
            elapsed = perf_counter() - started
        failed = bool(error) or (status or 0) >= 400
        gate.stats.errors += int(failed)
        logger.info(
            "url=%s status=%s bytes=%d elapsed=%.4f attempt=%d error=%s",
            url,
            status if status is not None else error,
            len(body),
            elapsed,
            attempt + 1,
            error or "-",
        )
        if retry and attempt + 1 < attempts:
            pause = max(
                retry_after(headers.get("Retry-After")),
                min(8.0, base_delay * 2**attempt) + random.uniform(0, base_delay),
            )
            # Retry-After стримує також інші задачі того самого хоста.
            host = gate.host(url)
            host.next_start = max(host.next_start, perf_counter() + pause)
            logger.info("url=%s retry_in=%.4f", url, pause)
            await asyncio.sleep(pause)
            continue
        return FetchResult(url, status, bytes(body), headers, elapsed, error)
    raise AssertionError("Недосяжна гілка")
