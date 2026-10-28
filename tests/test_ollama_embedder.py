import json
from collections.abc import Callable

import httpx
import numpy as np
import pytest

from codebase_qa.embeddings import Embedder, EmbedderUnavailableError, OllamaEmbedder

Handler = Callable[[httpx.Request], httpx.Response]


def fake_server(seen: list[list[str]]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/embed"
        assert body["model"] == "qwen3-embedding:0.6b"
        seen.append(body["input"])
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0, 0.0] for _ in body["input"]]})

    return handler


def embedder(handler: Handler, **kwargs: object) -> OllamaEmbedder:
    return OllamaEmbedder(
        "qwen3-embedding:0.6b",
        base_url="http://ollama.test/",
        transport=httpx.MockTransport(handler),
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


def test_probes_dimension_and_normalises() -> None:
    seen: list[list[str]] = []
    e = embedder(fake_server(seen))
    assert isinstance(e, Embedder)
    assert e.dimension == 3
    assert e.model_id == "qwen3-embedding:0.6b"
    vector = e.embed_query("q")
    assert vector.shape == (3,)
    np.testing.assert_allclose(vector, [0.6, 0.8, 0.0])


def test_query_prefix_applies_to_queries_only_and_documents_are_batched() -> None:
    seen: list[list[str]] = []
    e = embedder(fake_server(seen), query_prefix="Q: ", batch_size=2)
    seen.clear()
    e.embed_query("how")
    vectors = e.embed_documents(["a", "b", "c"])
    assert seen == [["Q: how"], ["a", "b"], ["c"]]
    assert vectors.shape == (3, 3)
    assert e.embed_documents([]).shape == (0, 3)


def test_missing_model_hints_pull() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(EmbedderUnavailableError, match=r"ollama pull qwen3-embedding:0\.6b"):
        embedder(handler)


def test_unreachable_server_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(EmbedderUnavailableError, match="cannot reach Ollama"):
        embedder(handler)


def test_count_mismatch_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": []})

    with pytest.raises(EmbedderUnavailableError, match="0 embeddings for 1 inputs"):
        embedder(handler)
