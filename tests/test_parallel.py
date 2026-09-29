"""Сумісність із лабораторною 4, однакові байти та безпечний spawn."""

import pickle
import subprocess
import sys
import sysconfig

import pytest

from findex.corpus import iter_documents
from findex.index import build_index
from findex.parallel import (
    BuildMetrics,
    WorkerError,
    build_parallel,
    build_partial,
    document_paths,
    merge,
)
from findex.store import load, save


def test_free_threaded_runtime():
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        assert not sys._is_gil_enabled()


@pytest.mark.parametrize("storage", ["plain", "slots", "array"])
@pytest.mark.parametrize("positions", [False, True])
def test_serial_matches_lab04(tiny_corpus, storage, positions):
    expected = build_index(iter_documents(tiny_corpus), storage, positions=positions)
    actual = merge(
        [
            build_partial(
                document_paths(tiny_corpus),
                tiny_corpus,
                storage=storage,
                positions=positions,
            )
        ]
    )
    assert actual == expected


def test_load_lab04_pickle(tiny_index, tmp_path):
    path = tmp_path / "old.bin"
    path.write_bytes(pickle.dumps({"version": 2, "index": tiny_index}, protocol=5))
    assert load(path) == tiny_index


@pytest.mark.slow
@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("storage", ["plain", "slots", "array"])
def test_identical_files(tiny_corpus, tmp_path, executor, storage):
    paths = document_paths(tiny_corpus)
    reference = build_parallel(paths, tiny_corpus, storage=storage, positions=True)
    metric = BuildMetrics()
    progress = []
    actual = build_parallel(
        paths,
        tiny_corpus,
        workers=3,
        executor=executor,
        storage=storage,
        positions=True,
        metrics=metric,
        progress=progress.append,
    )
    assert actual == reference
    assert sum(progress) == 4 and metric.merge_seconds >= 0
    for suffix in ("bin", "json"):
        a, b = tmp_path / f"a.{suffix}", tmp_path / f"b.{suffix}"
        save(reference, a)
        save(actual, b)
        assert a.read_bytes() == b.read_bytes()


def test_partial_picklable_and_merge_order(tiny_corpus):
    paths = document_paths(tiny_corpus)
    a = build_partial(paths[:2], tiny_corpus)
    b = build_partial(paths[2:], tiny_corpus, 2)
    before = pickle.dumps(a)
    assert merge([a, b]) == merge([pickle.loads(pickle.dumps(b)), a])
    assert pickle.dumps(a) == before
    with pytest.raises(ValueError, match="ID"):
        merge([a, a])
    with pytest.raises(ValueError, match="налаштування"):
        merge([a, build_partial(paths[2:], tiny_corpus, 2, positions=True)])
    assert merge([]).num_docs == 0


@pytest.mark.slow
@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_empty(tiny_corpus, executor):
    index = build_parallel(
        [], tiny_corpus, workers=2, executor=executor, positions=True
    )
    assert index.num_docs == 0 and index.has_positions


@pytest.mark.slow
@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_worker_error(tiny_corpus, executor):
    with pytest.raises(WorkerError) as error:
        build_parallel(
            [tiny_corpus / "missing.txt"], tiny_corpus, workers=2, executor=executor
        )
    assert isinstance(error.value.__cause__, FileNotFoundError)


@pytest.mark.slow
def test_spawn_cli_traceback_and_no_output_file(tmp_path):
    corpus = tmp_path / "bad"
    corpus.mkdir()
    (corpus / "bad.txt").write_bytes(b"\xff")
    target = tmp_path / "unchanged.bin"
    target.write_bytes(b"keep")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "findex",
            "index",
            str(corpus),
            "--out",
            str(target),
            "--executor",
            "processes",
            "--workers",
            "2",
        ],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert b"Traceback" in result.stderr and b"UnicodeDecodeError" in result.stderr
    assert target.read_bytes() == b"keep"


def test_invalid_parameters(tiny_corpus):
    with pytest.raises(ValueError):
        build_parallel([], tiny_corpus, workers=0)
    with pytest.raises(ValueError):
        build_parallel([], tiny_corpus, executor="bad")
    with pytest.raises(ValueError):
        document_paths(tiny_corpus, -1)
    with pytest.raises(ValueError):
        document_paths(tiny_corpus / "missing")
    with pytest.raises(ValueError):
        build_partial([], tiny_corpus, -1)
