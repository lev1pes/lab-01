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
from findex.models import ArrayPostings, DocMeta, Index, PlainPosting, Posting
from findex.timing import timed
from findex.tokenize import tokenize

STORAGES = ("plain", "slots", "array")


@timed
def build_index(
    documents: Iterable[Document], storage: str = "slots", *, positions=False
) -> Index:
    """Номер документа зростає, тому постінги одразу відсортовані."""
    if storage not in STORAGES:
        raise ValueError(f"Невідоме зберігання: {storage}")
    if storage == "array":
        postings = defaultdict(lambda: ArrayPostings(array("I"), array("I")))
    else:
        postings = defaultdict(list)
    lengths: dict[int, int] = {}
    metadata: dict[int, DocMeta] = {}
    record = PlainPosting if storage == "plain" else Posting
    offsets = defaultdict(dict) if positions else None
    texts = {}
    for doc_id, document in enumerate(documents):
        if positions:
            local_positions = defaultdict(list)
            counts = Counter()
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
            if storage == "array":
                postings[term].doc_ids.append(doc_id)
                postings[term].tfs.append(tf)
            else:
                postings[term].append(record(doc_id, tf))
    return Index(
        dict(postings),
        lengths,
        metadata,
        storage,
        dict(offsets) if positions else None,
        texts,
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
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
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
    print(f"Побудова: {built - started:.3f} с")
    print(f"Збереження: {finished - built:.3f} с")
    print(f"Загальний час: {finished - started:.3f} с")
    print(f"Пікова пам'ять: {peak / 1024**2:.3f} МіБ")
    print(f"Файл: {args.out} ({args.out.stat().st_size} байтів)")


if __name__ == "__main__":
    main()
