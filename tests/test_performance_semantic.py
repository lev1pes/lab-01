"""Числова еквівалентність, зберігання та семантика без мережі й моделі."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.index import build_index
from findex.models import NumpyPostings
from findex.scoring import BM25, TfIdf
from findex.search import SearchResult, search, search_reference
from findex.semantic import SemanticIndex, chunks, embed, normalize, retrieve, rrf
from findex.store import load, save
from findex.vector import top_indices
from findex.web import create_app, get_index


class FakeEncoder:
    """Детерміновані 384 виміри перевіряють математику, а не якість MiniLM."""

    def encode(self, texts):
        values = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            values[i, 0] = 1
            values[i, 1] = 3 if "async" in text or "concurrency" in text else 0
        return values


@pytest.mark.parametrize("scorer", [BM25(), BM25(k1=2, b=0), TfIdf()])
@pytest.mark.parametrize(
    "query",
    ["python", "NOT java", "python OR café", 'python AND "event loop"', "відсутнє"],
)
@pytest.mark.parametrize("k", [0, 1, 3, 20])
def test_ranking_matches_lab3(tiny_index, scorer, query, k):
    expected = search_reference(tiny_index, query, scorer, k)
    actual = search(tiny_index, query, scorer, k)
    assert [r.doc_id for r in actual] == [r.doc_id for r in expected]
    assert [r.score for r in actual] == pytest.approx(
        [r.score for r in expected], rel=1e-12
    )
    assert [r.snippet for r in actual] == [r.snippet for r in expected]


def test_ties_across_partition_boundary():
    ids = np.array([8, 2, 7, 1, 9], dtype=np.int32)
    scores = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    assert ids[top_indices(scores, ids, 3)].tolist() == [2, 1, 7]


def test_numpy_roundtrip(tiny_index, tmp_path):
    assert tiny_index.length_array.dtype == np.int32
    assert not tiny_index.length_array.flags.writeable
    assert all(isinstance(p, NumpyPostings) for p in tiny_index.numpy_postings.values())
    path = tmp_path / "index.json"
    save(tiny_index, path)
    restored = load(path)
    assert restored == tiny_index
    assert restored.storage == "numpy"
    assert all(p.tfs.dtype == np.int32 for p in restored.numpy_postings.values())


def test_chunks_overlap():
    assert list(chunks("a b c d e f", 4, 2)) == ["a b c d", "c d e f"]
    assert list(chunks("")) == []
    with pytest.raises(ValueError):
        list(chunks("abc", 1, 1))


def test_semantic_roundtrip_and_best_chunk(tiny_index, tmp_path):
    encoder = FakeEncoder()
    semantic = embed(tiny_index, encoder)
    semantic.save(tmp_path)
    restored = SemanticIndex.load(tmp_path, tiny_index)
    np.testing.assert_allclose(np.linalg.norm(restored.vectors, axis=1), 1)
    ranked = retrieve(
        tiny_index, "concurrency", mode="semantic", semantic=restored, encoder=encoder
    )
    assert ranked[0].title == "a"
    assert len({r.doc_id for r in ranked}) == len(ranked)
    hybrid = retrieve(
        tiny_index,
        "concurrency python",
        mode="hybrid",
        semantic=restored,
        encoder=encoder,
    )
    lexical = search(tiny_index, "concurrency OR python", BM25(), tiny_index.num_docs)
    expected = rrf([lexical, ranked])
    assert [r.doc_id for r in hybrid] == [r.doc_id for r in expected]
    assert all(r.snippet for r in hybrid)
    assert [r.score for r in hybrid] == pytest.approx([r.score for r in expected])
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["corpus_hash"] = "wrong"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="корпусу"):
        SemanticIndex.load(tmp_path, tiny_index)


@pytest.mark.parametrize("damage", ["dtype", "nan", "shape", "ids", "norm"])
def test_bad_embeddings(tiny_index, tmp_path, damage):
    semantic = embed(tiny_index, FakeEncoder())
    semantic.save(tmp_path)
    if damage == "ids":
        np.save(
            tmp_path / "doc_ids.npy",
            np.full(len(semantic.doc_ids), 999, dtype=np.int32),
        )
    else:
        array = semantic.vectors.copy()
        if damage == "dtype":
            array = array.astype(np.float64)
        elif damage == "nan":
            array[0, 0] = np.nan
        elif damage == "shape":
            array = array[:, :3]
        else:
            array *= 2
        np.save(tmp_path / "vectors.npy", array)
    with pytest.raises(ValueError):
        SemanticIndex.load(tmp_path, tiny_index)


def test_normalization_and_rrf():
    with pytest.raises(ValueError):
        normalize(np.zeros((2, 384), dtype=np.float32))
    a, b = SearchResult(1, 99, "a"), SearchResult(2, 0.1, "b")
    result = rrf([[a, b], [b]])
    assert result[0].doc_id == 2
    assert result[0].score == pytest.approx(1 / 62 + 1 / 61)
    with pytest.raises(ValueError):
        rrf([], 0)


def test_modes_in_api_and_ui(tiny_index):
    application = create_app()
    application.dependency_overrides[get_index] = lambda: tiny_index
    with TestClient(application) as client:
        assert (
            client.get("/search", params={"q": "hi", "mode": "semantic"}).status_code
            == 503
        )
        application.state.semantic = embed(tiny_index, FakeEncoder())
        application.state.encoder = FakeEncoder()
        response = client.get(
            "/search", params={"q": "concurrency", "mode": "semantic", "k": 1}
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["title"] == "a"
        html = client.get(
            "/", params={"q": "concurrency", "mode": "hybrid", "k": 1}
        ).text
        assert 'name="mode"' in html and "mode=hybrid" in html
        assert (
            client.get("/search", params={"q": "hi", "mode": "invalid"}).status_code
            == 422
        )


def test_embed_and_search_cli(index_path, tmp_path, monkeypatch):
    monkeypatch.setattr("findex.cli.MiniLM", lambda *a, **kw: FakeEncoder())
    directory = tmp_path / "embeddings"
    runner = CliRunner()
    result = runner.invoke(app, ["embed", str(index_path), "--out", str(directory)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "search",
            str(index_path),
            "concurrency",
            "--mode",
            "hybrid",
            "--embeddings",
            str(directory),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout.splitlines()[0])["title"] == "a"


def test_empty_semantic(tiny_corpus, tmp_path):
    index = build_index([])
    semantic = embed(index, FakeEncoder())
    semantic.save(tmp_path)
    restored = SemanticIndex.load(tmp_path, index)
    assert restored.rank(index, "q", FakeEncoder()) == []


def test_benchmark_search(benchmark, tiny_index):
    benchmark(search, tiny_index, "python OR café")


def test_benchmark_build(benchmark, tiny_corpus):
    def run():
        index = build_index(iter_documents(tiny_corpus), positions=True)
        index.close()

    benchmark(run)


def test_checkpoint_resume(tiny_index, tmp_path):
    class CountingEncoder(FakeEncoder):
        calls = 0

        def encode(self, texts):
            self.calls += 1
            return super().encode(texts)

    encoder = CountingEncoder()
    first = embed(tiny_index, encoder, checkpoint_dir=tmp_path)
    calls = encoder.calls
    second = embed(tiny_index, encoder, checkpoint_dir=tmp_path)
    assert encoder.calls == calls
    np.testing.assert_allclose(first.vectors, second.vectors)
    part = next(tmp_path.glob("*.npy"))
    np.save(part, np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="checkpoint"):
        embed(tiny_index, encoder, checkpoint_dir=tmp_path)
