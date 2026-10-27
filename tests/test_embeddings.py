import numpy as np
import pytest

from codebase_qa.embeddings import PRESETS, Embedder, HashEmbedder, SentenceTransformerEmbedder


def test_hash_embedder_is_deterministic_and_unit_length() -> None:
    embedder = HashEmbedder(dimension=32)
    first = embedder.embed_documents(["def load_config(path): ...", "class Parser: ..."])
    second = embedder.embed_documents(["def load_config(path): ...", "class Parser: ..."])

    assert first.shape == (2, 32)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-6)


def test_hash_embedder_places_shared_vocabulary_closer() -> None:
    embedder = HashEmbedder()
    query = embedder.embed_query("load config file")
    related, unrelated = embedder.embed_documents(
        ["def load_config(file): read the config file", "class HttpRetryPolicy: backoff"]
    )

    assert float(query @ related) > float(query @ unrelated)


def test_hash_embedder_counts_document_embeddings() -> None:
    embedder = HashEmbedder()
    embedder.embed_documents(["a", "b"])
    embedder.embed_query("c")

    assert embedder.documents_embedded == 2


def test_hash_embedder_satisfies_protocol() -> None:
    assert isinstance(HashEmbedder(), Embedder)


def test_presets_pin_revisions() -> None:
    for preset in PRESETS.values():
        assert preset.revision is not None
        assert len(preset.revision) == 40


@pytest.mark.model
def test_minilm_embeds_unit_vectors() -> None:
    embedder = SentenceTransformerEmbedder.from_preset("minilm", device="cpu")
    vectors = embedder.embed_documents(["def add(a, b): return a + b"])

    assert embedder.dimension == 384
    assert embedder.max_seq_length == 256
    assert vectors.shape == (1, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


@pytest.mark.model
def test_coderankembed_prefixes_queries_only() -> None:
    embedder = SentenceTransformerEmbedder.from_preset("coderankembed", device="cpu")
    text = "parse the configuration file"
    as_document = embedder.embed_documents([text])[0]
    as_query = embedder.embed_query(text)

    assert embedder.dimension == 768
    np.testing.assert_allclose(np.linalg.norm(as_query), 1.0, rtol=1e-5)
    assert not np.allclose(as_document, as_query)
