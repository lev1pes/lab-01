FROM ghcr.io/astral-sh/uv:0.12.14 AS uv
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
COPY scripts/download_corpus.py ./scripts/download_corpus.py
RUN .venv/bin/python scripts/download_corpus.py && \
    .venv/bin/findex index data/python-docs --out data/web-index.json --positions --executor serial --workers 1

FROM python:3.12-slim-bookworm AS runtime
RUN groupadd --gid 10001 findex && useradd --uid 10001 --gid findex --create-home findex
WORKDIR /app
COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/data/web-index.json /app/data/web-index.json
ENV PATH="/app/.venv/bin:$PATH" INDEX_PATH=/app/data/web-index.json HOST=0.0.0.0 PORT=8000 WORKERS=1 PYTHONUNBUFFERED=1
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=4)"
CMD ["findex", "serve"]
