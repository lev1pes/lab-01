"""Одне асинхронне читання robots.txt на origin протягом обходу."""

import asyncio
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT, RequestGate, fetch
from findex.crawler.urls import origin


class RobotsCache:
    def __init__(
        self, client: httpx.AsyncClient, gate: RequestGate, timeout: float = 10.0
    ) -> None:
        self.client = client
        self.gate = gate
        self.timeout = timeout
        self.rules: dict[str, RobotFileParser] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        key = origin(url)
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self.rules:
                result = await fetch(
                    self.client,
                    key + "/robots.txt",
                    gate=self.gate,
                    timeout=self.timeout,
                    max_bytes=512 * 1024,
                )
                parser = RobotFileParser()
                if result.status == 404 and not result.error:
                    parser.parse(["User-agent: *", "Allow: /"])
                elif result.status == 200 and not result.error:
                    parser.parse(result.body.decode("utf-8", "replace").splitlines())
                else:
                    # 401/403, недоступність і редірект: закрита політика.
                    parser.parse(["User-agent: *", "Disallow: /"])
                self.rules[key] = parser
                delay = parser.crawl_delay(USER_AGENT) or 0
                self.gate.host(url).delay = max(self.gate.delay, float(delay))
        return self.rules[key].can_fetch(USER_AGENT, url)
