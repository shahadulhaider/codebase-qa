from pathlib import Path

import numpy as np
import pytest

from codebase_qa.embeddings import HashEmbedder
from codebase_qa.vector_index import ChromaVectorIndex, HnswConfig, IndexConfigError

DOCS = {
    "load": ("def load_config(path): read config file", {"language": "python", "start_line": 1}),
    "retry": ("class RetryPolicy: exponential backoff", {"language": "python", "start_line": 10}),
    "fetch": (
        "function fetchUser(id) { return http.get(id) }",
        {"language": "typescript", "start_line": 3},
    ),
}


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


def _populate(index: ChromaVectorIndex, embedder: HashEmbedder) -> None:
    ids = list(DOCS)
    documents = [DOCS[i][0] for i in ids]
    index.upsert(ids, embedder.embed_documents(documents), documents, [DOCS[i][1] for i in ids])


def test_query_ranks_by_cosine_distance(tmp_path: Path, embedder: HashEmbedder) -> None:
    index = ChromaVectorIndex.open(tmp_path, "chunks")
    _populate(index, embedder)
    query = embedder.embed_query("load the config file")

    hits = index.query(query, k=3)

    assert hits[0].id == "load"
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)
    stored = embedder.embed_documents([DOCS["load"][0]])[0]
    assert hits[0].distance == pytest.approx(1.0 - float(query @ stored), abs=1e-5)
    assert hits[0].metadata == {"language": "python", "start_line": 1}


def test_query_applies_metadata_filter(tmp_path: Path, embedder: HashEmbedder) -> None:
    index = ChromaVectorIndex.open(tmp_path, "chunks")
    _populate(index, embedder)

    hits = index.query(embedder.embed_query("load config"), k=3, where={"language": "typescript"})

    assert [h.id for h in hits] == ["fetch"]


def test_upsert_delete_and_get_embeddings(tmp_path: Path, embedder: HashEmbedder) -> None:
    index = ChromaVectorIndex.open(tmp_path, "chunks")
    _populate(index, embedder)
    index.delete(["retry"])

    vectors = index.get_embeddings(["load", "retry"])

    assert index.count() == 2
    assert set(vectors) == {"load"}
    np.testing.assert_allclose(
        vectors["load"], embedder.embed_documents([DOCS["load"][0]])[0], rtol=1e-6
    )


def test_collection_persists_across_clients(tmp_path: Path, embedder: HashEmbedder) -> None:
    _populate(ChromaVectorIndex.open(tmp_path, "chunks"), embedder)

    assert ChromaVectorIndex.open(tmp_path, "chunks").count() == 3


def test_reopening_with_different_space_is_rejected(tmp_path: Path) -> None:
    ChromaVectorIndex.open(tmp_path, "chunks", HnswConfig(space="cosine"))

    with pytest.raises(IndexConfigError, match="space"):
        ChromaVectorIndex.open(tmp_path, "chunks", HnswConfig(space="l2"))


def test_reopening_with_different_ef_search_updates_it(tmp_path: Path) -> None:
    ChromaVectorIndex.open(tmp_path, "chunks", HnswConfig(ef_search=100))

    assert ChromaVectorIndex.open(tmp_path, "chunks", HnswConfig(ef_search=40)).ef_search == 40


def test_upsert_rejects_mismatched_lengths(tmp_path: Path, embedder: HashEmbedder) -> None:
    index = ChromaVectorIndex.open(tmp_path, "chunks")

    with pytest.raises(ValueError, match="length"):
        index.upsert(["a", "b"], embedder.embed_documents(["x"]), ["x"], [{}])
