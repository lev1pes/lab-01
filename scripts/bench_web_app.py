"""Лише для локального досліду: навмисно блокувальний async-маршрут."""

import asyncio
from contextlib import asynccontextmanager, suppress
from time import perf_counter

from findex.web import (
    IndexDependency,
    ParamsDependency,
    create_app,
    execute_search,
)
from findex.web.schemas import SearchResponse


def measured_app(blocking=False):
    app = create_app()
    original = app.router.lifespan_context
    if blocking:
        app.router.routes[:] = [
            r for r in app.router.routes if getattr(r, "path", "") != "/search"
        ]

        @app.get("/search", response_model=SearchResponse)
        async def bad_search(params: ParamsDependency, index: IndexDependency):
            return execute_search(index, params)

    @asynccontextmanager
    async def lifespan(application):
        async with original(application):
            application.state.max_lag = 0.0

            async def ticker():
                while True:
                    start = perf_counter()
                    await asyncio.sleep(0.01)
                    application.state.max_lag = max(
                        application.state.max_lag, perf_counter() - start - 0.01
                    )

            task = asyncio.create_task(ticker())
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app.router.lifespan_context = lifespan

    @app.get("/__bench/lag")
    async def lag(reset: bool = False):
        value = app.state.max_lag * 1000
        if reset:
            app.state.max_lag = 0.0
        return {"max_lag_ms": value}

    return app


def make_async():
    return measured_app(True)


def make_sync():
    return measured_app()
