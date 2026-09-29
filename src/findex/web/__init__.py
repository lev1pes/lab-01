"""HTTP API та серверна HTML-сторінка findex."""

import asyncio
import logging
import math
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast
from urllib.parse import urlencode
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex.models import Index
from findex.scoring import BM25, TfIdf
from findex.search import search
from findex.semantic import Encoder, MiniLM, SemanticIndex, retrieve
from findex.store import load
from findex.web.schemas import (
    DocumentOut,
    HealthOut,
    ResultOut,
    SearchParams,
    SearchResponse,
    StatsOut,
)
from findex.web.settings import Settings, get_settings

logger = logging.getLogger("uvicorn.error")
ROOT = Path(__file__).parent
templates = Jinja2Templates(directory=ROOT / "templates")


def highlight(value: str) -> Markup:
    """Екранувати весь текст перед додаванням власних тегів mark."""
    chunks = re.split(r"(\[[^\[\]]+\])", value)
    return Markup("").join(
        Markup("<mark>{}</mark>").format(escape(part[1:-1]))
        if part.startswith("[") and part.endswith("]")
        else escape(part)
        for part in chunks
    )


templates.env.filters["highlight"] = highlight


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.index = None
    app.state.semantic = None
    app.state.encoder = None
    app.state.started = perf_counter()
    app.state.index_bytes = 0
    override = app.dependency_overrides.get(get_settings)
    settings = cast(Settings, override()) if override else get_settings()
    # Тести підставляють індекс через Depends, тому диск їм не потрібен.
    if get_index not in app.dependency_overrides:
        try:
            index = await asyncio.to_thread(load, settings.index_path)
            app.state.index = index
            app.state.index_bytes = settings.index_path.stat().st_size
            logger.info("index_loaded documents=%d", index.num_docs)
            if settings.embeddings_path is not None:
                app.state.semantic = await asyncio.to_thread(
                    SemanticIndex.load, settings.embeddings_path, index
                )
                app.state.encoder = await asyncio.to_thread(
                    MiniLM, cache_dir=settings.model_cache
                )
        except (OSError, ValueError):
            logger.exception("index_unavailable")
    try:
        yield
    finally:
        index = app.state.index
        if index is not None:
            index.close()


def get_index(request: Request) -> Index:
    index = cast(Index | None, getattr(request.app.state, "index", None))
    if index is None or index.closed:
        raise HTTPException(503, "Індекс ще не завантажено")
    return index


IndexDependency = Annotated[Index, Depends(get_index)]
ParamsDependency = Annotated[SearchParams, Query()]


def execute_search(
    index: Index,
    params: SearchParams,
    semantic: SemanticIndex | None = None,
    encoder: Encoder | None = None,
) -> SearchResponse:
    started = perf_counter()
    try:
        if params.mode != "keyword":
            if semantic is None or encoder is None:
                raise HTTPException(503, "Семантичний індекс ще не завантажено")
            ranked_all = retrieve(
                index,
                params.q,
                k=index.num_docs,
                mode=params.mode,
                semantic=semantic,
                encoder=encoder,
            )
            total = len(ranked_all)
            start = (params.page - 1) * params.k
            return SearchResponse(
                query=params.q,
                total=total,
                took_ms=round((perf_counter() - started) * 1000, 3),
                page=params.page,
                pages=math.ceil(total / params.k),
                results=[
                    ResultOut.model_validate(r)
                    for r in ranked_all[start : start + params.k]
                ],
            )
        total = len(index.matched_ids(params.q))
        end = min(total, params.page * params.k)
        start = (params.page - 1) * params.k
        ranked = (
            search(index, params.q, BM25() if params.scorer == "bm25" else TfIdf(), end)
            if start < total
            else []
        )
    except (ValueError, RecursionError) as error:
        detail = str(error) if isinstance(error, ValueError) else "Запит надто складний"
        raise HTTPException(422, detail) from None
    return SearchResponse(
        query=params.q,
        total=total,
        took_ms=round((perf_counter() - started) * 1000, 3),
        page=params.page,
        pages=math.ceil(total / params.k),
        results=[ResultOut.model_validate(result) for result in ranked[start:end]],
    )


def document(index: Index, doc_id: int) -> DocumentOut:
    if doc_id not in index.doc_meta:
        raise HTTPException(404, "Документ не знайдено")
    meta = index.doc_meta[doc_id]
    return DocumentOut(
        doc_id=doc_id,
        title=meta.title,
        path=meta.path,
        text=index.texts.get(doc_id, ""),
    )


def create_app() -> FastAPI:
    application = FastAPI(title="findex", version="1.0.0", lifespan=lifespan)
    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @application.middleware("http")
    async def request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid4().hex
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_id=%s unexpected_error", request_id)
            response = JSONResponse(
                {"detail": "Внутрішня помилка сервера", "request_id": request_id},
                status_code=500,
            )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        logger.info(
            "request_id=%s method=%s path=%s status=%d elapsed_ms=%.3f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (perf_counter() - started) * 1000,
        )
        return response

    @application.get("/search", response_model=SearchResponse)
    def search_route(
        request: Request, params: ParamsDependency, index: IndexDependency
    ) -> SearchResponse:
        return execute_search(
            index,
            params,
            getattr(request.app.state, "semantic", None),
            getattr(request.app.state, "encoder", None),
        )

    @application.get("/docs/{doc_id}", response_model=DocumentOut)
    def doc_api(doc_id: int, index: IndexDependency) -> DocumentOut:
        return document(index, doc_id)

    @application.get("/stats", response_model=StatsOut)
    def stats(request: Request, index: IndexDependency) -> StatsOut:
        return StatsOut(
            documents=index.num_docs,
            vocabulary=len(index),
            tokens=sum(index.doc_lengths.values()),
            index_bytes=int(getattr(request.app.state, "index_bytes", 0)),
            uptime_seconds=perf_counter()
            - float(getattr(request.app.state, "started", perf_counter())),
        )

    @application.get("/health", response_model=HealthOut)
    async def health(index: IndexDependency) -> HealthOut:
        return HealthOut()

    @application.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        index: IndexDependency,
        q: str = "",
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Annotated[str, Query(pattern="^(bm25|tfidf)$")] = "bm25",
        page: Annotated[int, Query(ge=1, le=10000)] = 1,
        mode: Annotated[str, Query(pattern="^(keyword|semantic|hybrid)$")] = "keyword",
    ) -> Response:
        result = None
        error = ""
        if q:
            try:
                params = SearchParams.model_validate(
                    {"q": q, "k": k, "scorer": scorer, "page": page, "mode": mode}
                )
                result = execute_search(
                    index,
                    params,
                    getattr(request.app.state, "semantic", None),
                    getattr(request.app.state, "encoder", None),
                )
            except (ValueError, HTTPException) as exc:
                error = (
                    str(exc.detail)
                    if isinstance(exc, HTTPException)
                    else "Перевірте запит (до 300 символів)."
                )

        def link(number: int) -> str:
            return "/?" + urlencode(
                {"q": q, "k": k, "scorer": scorer, "page": number, "mode": mode}
            )

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "q": q,
                "scorer": scorer,
                "mode": mode,
                "k": k,
                "result": result,
                "error": error,
                "count": index.num_docs,
                "previous": link(page - 1),
                "next": link(page + 1),
            },
        )

    @application.get("/doc/{doc_id}", response_class=HTMLResponse)
    def doc_page(request: Request, doc_id: int, index: IndexDependency) -> Response:
        return templates.TemplateResponse(
            request, "document.html", {"doc": document(index, doc_id)}
        )

    return application


app = create_app()
