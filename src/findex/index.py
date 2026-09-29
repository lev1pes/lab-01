"""Побудова індексу одним проходом по документах лабораторної 1."""

import argparse
import logging
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.models import (
    ArrayPostings,
    DocMeta,
    Index,
    IndexState,
    PlainPosting,
    Posting,
    PostingList,
    Storage,
)
from findex.timing import timed
from findex.tokenize import tokenize

STORAGES: tuple[Storage, ...] = ("plain", "slots", "array")


@timed
def build_index(
    documents: Iterable[Document],
    storage: Storage = "slots",
    *,
    positions: bool = False,
) -> Index:
    return Index(**build_state(documents, storage, positions=positions))


def build_state(
    documents: Iterable[Document],
    storage: Storage = "slots",
    *,
    positions: bool = False,
    start_id: int = 0,
) -> IndexState:
    """Номер документа зростає, тому постінги одразу відсортовані."""
    if storage not in STORAGES:
        raise ValueError(f"Невідоме зберігання: {storage}")

    def new_postings() -> PostingList:
        return ArrayPostings(array("I"), array("I")) if storage == "array" else []

    postings: defaultdict[str, PostingList] = defaultdict(new_postings)
    lengths: dict[int, int] = {}
    metadata: dict[int, DocMeta] = {}
    record = PlainPosting if storage == "plain" else Posting
    offsets: defaultdict[str, dict[int, tuple[int, ...]]] | None = (
        defaultdict(dict) if positions else None
    )
    texts: dict[int, str] = {}
    for doc_id, document in enumerate(documents, start_id):
        if offsets is not None:
            local_positions: defaultdict[str, list[int]] = defaultdict(list)
            counts: Counter[str] = Counter()
            for offset, term in enumerate(tokenize(document.text)):
                counts[term] += 1
                local_positions[term].append(offset)
            for term, values in local_positions.items():
                offsets[term][doc_id] = tuple(values)
        else:
            counts = Counter(tokenize(document.text))
        texts[doc_id] = document.text
        lengths[doc_id] = counts.total()
        # Назва файла зрозуміліша за перший рядок розмітки документації.
        title = document.path.stem.replace("_", " ")
        metadata[doc_id] = DocMeta(document.doc_id, title)
        for term, tf in counts.items():
            items = postings[term]
            if isinstance(items, ArrayPostings):
                items.doc_ids.append(doc_id)
                items.tfs.append(tf)
            else:
                items.append(record(doc_id, tf))
    return IndexState(
        postings=dict(postings),
        doc_lengths=lengths,
        doc_meta=metadata,
        storage=storage,
        positions=dict(offsets) if offsets is not None else None,
        texts=texts,
    )


def main(argv: Sequence[str] | None = None) -> None:
    from findex.store import save

    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу")
    parser.add_argument("root", type=Path, help="каталог корпусу")
    parser.add_argument("--out", required=True, type=Path, help="файл індексу")
    parser.add_argument("--storage", choices=STORAGES, default="slots")
    parser.add_argument("--format", choices=("pickle", "json"), default=None)
    parser.add_argument(
        "--positions", action="store_true", help="зберігати позиції для фраз"
    )
    parser.add_argument("--verbose", action="store_true", help="показувати журнал часу")
    args = parser.parse_args(argv)
    if __name__ == "__main__":
        from findex.cli import configure_logging

        configure_logging(2 if getattr(args, "verbose", False) else 1)
    tracemalloc.start()
    started = time.perf_counter()
    try:
        index = build_index(
            iter_documents(args.root), args.storage, positions=args.positions
        )
        built = time.perf_counter()
        save(index, args.out, args.format)
        finished = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))
    finally:
        tracemalloc.stop()
    print(f"Документів: {len(index.doc_meta)}")
    print(f"Токенів: {sum(index.doc_lengths.values())}")
    print(f"Термінів: {len(index.postings)}")
    log = logging.getLogger("findex")
    log.info("Побудова: %.3f с", built - started)
    log.info("Збереження: %.3f с", finished - built)
    log.info("Загальний час: %.3f с", finished - started)
    log.info("Пікова пам'ять: %.3f МіБ", peak / 1024**2)
    print(f"Файл: {args.out} ({args.out.stat().st_size} байтів)")


if __name__ == "__main__":
    main()
