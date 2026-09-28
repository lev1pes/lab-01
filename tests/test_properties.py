"""Властивості на згенерованих корпусах і відсортованих множинах ID."""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index
from findex.search import merge_and, merge_not, merge_or
from findex.store import load, save

texts = st.lists(st.text(alphabet="abc Кітдімé’-'012\n", max_size=80), max_size=8)
ids = st.lists(st.integers(min_value=0, max_value=1000), unique=True).map(sorted)


def documents(texts):
    return (Document(str(i), Path(f"{i}.txt"), text) for i, text in enumerate(texts))


@given(ids, ids)
def test_merge_matches_sets(a, b):
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


@given(texts, st.sampled_from(["plain", "slots", "array"]))
def test_postings_sorted(texts, storage):
    index = build_index(documents(texts), storage, positions=True)
    for postings in index.values():
        doc_ids = [p.doc_id for p in postings]
        assert doc_ids == sorted(set(doc_ids))
        assert all(p.tf > 0 for p in postings)
    assert sum(index.doc_lengths.values()) == sum(
        p.tf for ps in index.values() for p in ps
    )


@pytest.mark.slow
@settings(max_examples=60, deadline=None)
@given(
    texts,
    st.sampled_from(["plain", "slots", "array"]),
    st.booleans(),
    st.sampled_from(["pickle", "json"]),
)
def test_save_load_roundtrip(tmp_path_factory, texts, storage, positions, format):
    # Фабрика має session scope, але каталог та індекс нові для кожного прикладу.
    path = tmp_path_factory.mktemp("roundtrip") / "index.bin"
    index = build_index(documents(texts), storage, positions=positions)
    save(index, path, format)
    restored = load(path, format)
    assert restored == index
    assert restored.has_positions == positions
