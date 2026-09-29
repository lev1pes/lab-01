"""JSONL і журнал записуються поза потоком циклу подій."""

import asyncio
import json
import logging
from collections.abc import Callable, Generator
from contextlib import aclosing, contextmanager
from dataclasses import asdict
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import SimpleQueue
from typing import TextIO

import httpx

from findex.crawler.crawl import Page, crawl
from findex.crawler.fetch import CrawlStats, logger


@contextmanager
def crawl_log(path: Path) -> Generator[None, None, None]:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    queue: SimpleQueue[logging.LogRecord] = SimpleQueue()
    queued = QueueHandler(queue)
    listener = QueueListener(queue, handler)
    old_level, old_propagate = logger.level, logger.propagate
    logger.addHandler(queued)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    listener.start()
    try:
        yield
    finally:
        listener.stop()
        logger.removeHandler(queued)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        handler.close()


def write_page(target: TextIO, page: Page) -> None:
    target.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
    target.flush()


async def collect(
    seed: str,
    target: TextIO,
    *,
    stats: CrawlStats,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    timeout: float,
    update: Callable[[CrawlStats], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    async def report() -> None:
        while True:
            if update is not None:
                await asyncio.to_thread(update, stats)
            await asyncio.sleep(0.1)

    async with asyncio.TaskGroup() as group:
        reporter = group.create_task(report())
        try:
            async with aclosing(
                crawl(
                    [seed],
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host=per_host,
                    delay=delay,
                    timeout=timeout,
                    stats=stats,
                    transport=transport,
                )
            ) as pages:
                async for page in pages:
                    await asyncio.to_thread(write_page, target, page)
        finally:
            reporter.cancel()
    if update is not None:
        await asyncio.to_thread(update, stats)
