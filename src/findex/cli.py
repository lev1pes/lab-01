"""Встановлювана команда: дані у stdout, журнал і прогрес у stderr."""

import asyncio
import json
import logging
import re
import sys
import tracemalloc
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from findex.corpus import iter_documents
from findex.crawler.fetch import CrawlStats
from findex.crawler.output import collect, crawl_log
from findex.crawler.urls import canonicalize
from findex.index import build_index
from findex.parallel import WorkerError, build_parallel, document_paths
from findex.scoring import BM25, Scorer, TfIdf
from findex.semantic import MODEL, MiniLM, SemanticIndex, embed, retrieve
from findex.store import open_index, save

app = typer.Typer(
    help="findex — пошук у власному текстовому корпусі.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
log = logging.getLogger("findex")


def configure_logging(verbosity: int = 0) -> None:
    """Повторний виклик у CliRunner не накопичує обробники журналу."""
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(
        logging.DEBUG
        if verbosity >= 2
        else logging.INFO
        if verbosity
        else logging.WARNING
    )
    log.propagate = False


@app.callback()
def options(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="-v: час і пам'ять; -vv: детальний журнал.",
        ),
    ] = 0,
) -> None:
    configure_logging(verbose)


@contextmanager
def operation(trace_memory: bool = True) -> Generator[None, None, None]:
    """Спільна межа очікуваних помилок і вимірювання ресурсів CLI."""
    started = perf_counter()
    if trace_memory:
        tracemalloc.start()
    try:
        yield
    except WorkerError:
        log.exception("Помилка воркера; індекс не збережено")
        raise typer.Exit(code=1) from None
    except (OSError, ValueError, OverflowError) as error:
        if isinstance(error, FileNotFoundError):
            message = f"Файл або каталог не знайдено: {error.filename}"
        else:
            message = " ".join(str(error).splitlines())
        log.error("Помилка: %s", message)
        raise typer.Exit(code=1) from None
    finally:
        if trace_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            log.info(
                "Час: %.3f с; пікова пам'ять: %.3f МіБ",
                perf_counter() - started,
                peak / 1024**2,
            )
        else:
            log.info("Час: %.3f с", perf_counter() - started)


@app.command("index")
def index_command(
    corpus: Annotated[Path, typer.Argument(help="Каталог .txt або файл crawl.jsonl.")],
    out: Annotated[
        Path, typer.Option("--out", help="Файл індексу; .json обирає JSON.")
    ],
    positions: Annotated[
        bool, typer.Option("--positions", help="Позиції для фразового пошуку.")
    ] = False,
    limit: Annotated[
        int | None, typer.Option(min=0, help="Обмежити кількість документів.")
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Кількість воркерів.")] = 4,
    executor: Annotated[
        Literal["serial", "threads", "processes"] | None,
        typer.Option(help="Для .txt: processes; для JSONL: serial."),
    ] = None,
) -> None:
    """Побудувати та зберегти інвертований індекс."""
    # Tracemalloc лише в батьку несправедливо сповільнював би serial/threads.
    # RSS усіх конфігурацій окремо міряє benchmark_concurrency.py.
    with operation(trace_memory=False):
        jsonl = corpus.is_file() and corpus.suffix == ".jsonl"
        if not corpus.is_dir() and not jsonl:
            raise ValueError(f"Каталог корпусу не знайдено: {corpus}")
        if jsonl and executor not in (None, "serial"):
            raise ValueError("Потоковий JSONL підтримує --executor serial")
        paths = [] if jsonl else document_paths(corpus, limit)
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("{task.description}: {task.completed:.0f} документів"),
            console=Console(stderr=True),
            transient=False,
        ) as progress:
            task = progress.add_task("Індексація", total=None if jsonl else len(paths))
            if jsonl:
                documents = iter_documents(corpus)
                try:
                    index = build_index(islice(documents, limit), positions=positions)
                finally:
                    documents.close()
                progress.update(task, completed=index.num_docs)
            else:
                index = build_parallel(
                    paths,
                    corpus,
                    workers=workers,
                    executor=executor or "processes",
                    positions=positions,
                    progress=lambda n: progress.advance(task, n),
                )
        try:
            save(index, out)
            Console().print(
                f"Збережено: {out}; документів: {index.num_docs}; "
                f"термінів: {len(index)}",
                markup=False,
            )
        finally:
            index.close()


def crawl_table(stats: CrawlStats) -> Table:
    table = Table("Сторінок", "Стор./с", "Запитів у польоті", "Помилок", "Черга")
    table.add_row(
        str(stats.pages),
        f"{stats.rate:.2f}",
        str(stats.in_flight),
        str(stats.errors),
        str(stats.queued),
    )
    return table


@app.command("crawl")
def crawl_command(
    seed: Annotated[
        str, typer.Argument(help="Початкова HTTP(S)-адреса дозволеного сайту.")
    ],
    out: Annotated[Path, typer.Option(help="Потоковий JSONL.")] = Path(
        "data/crawl.jsonl"
    ),
    max_pages: Annotated[int, typer.Option(min=0)] = 500,
    concurrency: Annotated[int, typer.Option(min=1)] = 10,
    per_host: Annotated[int, typer.Option(min=1)] = 2,
    delay: Annotated[
        float, typer.Option(min=0, help="Пауза між стартами на хості.")
    ] = 0.2,
    timeout: Annotated[float, typer.Option(min=0.001)] = 10.0,
    log_path: Annotated[Path, typer.Option("--log", help="Статус кожного URL.")] = Path(
        "crawl.log"
    ),
    debug: Annotated[bool, typer.Option(help="Діагностика asyncio.")] = False,
) -> None:
    """Зібрати сторінки цього домену; індексація запускається окремою командою."""
    with operation(trace_memory=False):
        seed = canonicalize(seed)
        if out.resolve() == log_path.resolve():
            raise ValueError("JSONL і журнал мають бути різними файлами")
        out.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stats = CrawlStats()
        with crawl_log(log_path), out.open("w", encoding="utf-8") as target:
            with Live(
                crawl_table(stats), console=Console(stderr=True), auto_refresh=False
            ) as live:
                asyncio.run(
                    collect(
                        seed,
                        target,
                        stats=stats,
                        max_pages=max_pages,
                        concurrency=concurrency,
                        per_host=per_host,
                        delay=delay,
                        timeout=timeout,
                        update=lambda state: live.update(
                            crawl_table(state), refresh=True
                        ),
                    ),
                    debug=debug,
                )
        Console().print(f"Збережено {stats.pages} сторінок: {out}", markup=False)


@app.command("serve")
def serve_command(
    port: Annotated[int | None, typer.Option(min=1, max=65535)] = None,
    workers: Annotated[int | None, typer.Option(min=1, max=16)] = None,
) -> None:
    """Відкрити HTTP API та сторінку пошуку; конфіг — зі змінних оточення."""
    import uvicorn

    from findex.web.settings import get_settings

    with operation(trace_memory=False):
        settings = get_settings()
        uvicorn.run(
            "findex.web:app",
            host=settings.host,
            port=port or settings.port,
            workers=workers or settings.workers,
        )


def highlighted(text: str) -> Text:
    """Текст корпусу не виконується як Rich-розмітка."""
    result = Text(text)
    for match in re.finditer(r"\[[^\[\]]+\]", text):
        result.stylize("bold yellow", match.start(), match.end())
    return result


@app.command("search")
def search_command(
    index_path: Annotated[
        Path, typer.Argument(help="Власний довірений індекс (pickle або JSON).")
    ],
    query: Annotated[str, typer.Argument(help="Терміни, AND/OR/NOT, дужки, фрази.")],
    k: Annotated[int, typer.Option("--k", min=0, help="Кількість результатів.")] = 10,
    scorer: Annotated[
        Literal["bm25", "tfidf"], typer.Option(help="Формула оцінки.")
    ] = "bm25",
    mode: Annotated[
        Literal["keyword", "semantic", "hybrid"], typer.Option()
    ] = "keyword",
    embeddings: Annotated[Path, typer.Option()] = Path("data/embeddings"),
    model_cache: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="JSON Lines: один результат на рядок.")
    ] = False,
) -> None:
    """Знайти документи й показати ранжовані снипети."""
    with operation(), open_index(index_path) as index:
        # Pyright перевіряє структурну сумісність обох класів із Protocol.
        algorithm: Scorer = BM25() if scorer == "bm25" else TfIdf()
        semantic = SemanticIndex.load(embeddings, index) if mode != "keyword" else None
        encoder = MiniLM(cache_dir=model_cache) if semantic else None
        results = retrieve(
            index, query, algorithm, k, mode=mode, semantic=semantic, encoder=encoder
        )
        if json_output:
            for result in results:
                print(
                    json.dumps(
                        {
                            "doc_id": result.doc_id,
                            "score": result.score,
                            "title": result.title,
                            "path": index.doc_meta[result.doc_id].path,
                            "snippet": result.snippet,
                        },
                        ensure_ascii=False,
                    )
                )
        else:
            table = Table(title=f"Знайдено документів: {len(results)}")
            for column in ("ID", "Бал", "Документ", "Снипет"):
                table.add_column(column)
            for result in results:
                table.add_row(
                    str(result.doc_id),
                    f"{result.score:.5f}",
                    Text(index.doc_meta[result.doc_id].path),
                    highlighted(result.snippet),
                )
            Console().print(table)


@app.command("stats")
def stats_command(
    index_path: Annotated[Path, typer.Argument(help="Власний довірений індекс.")],
) -> None:
    """Показати кількість документів, токенів, словник і топ-50 термінів."""
    with operation(), open_index(index_path) as index:
        counts = Counter(
            {term: sum(p.tf for p in postings) for term, postings in index.items()}
        )
        summary = Table("Показник", "Значення")
        for label, value in (
            ("Документів", index.num_docs),
            ("Токенів", counts.total()),
            ("Термінів", len(index)),
            ("Середня довжина", round(index.avg_doc_length, 2)),
        ):
            summary.add_row(label, str(value))
        top = Table("Термін", "Частота", title="Топ-50")
        for term, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :50
        ]:
            top.add_row(Text(term), str(count))
        console = Console()
        console.print(summary)
        console.print(top)


@app.command("embed")
def embed_command(
    index_path: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option()] = Path("data/embeddings"),
    model: Annotated[str, typer.Option()] = MODEL,
    model_cache: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Порахувати нормалізовані MiniLM-вектори фрагментів і зберегти .npy."""
    with operation(trace_memory=False), open_index(index_path) as index:
        encoder = MiniLM(model, model_cache)
        with Progress(console=Console(stderr=True)) as progress:
            task = progress.add_task("Ембеддинги", total=None)
            semantic = embed(index, encoder, lambda n: progress.advance(task, n))
        semantic.save(out)
        Console().print(
            f"Збережено {len(semantic.doc_ids)} фрагментів: {out}", markup=False
        )
