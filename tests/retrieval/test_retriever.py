from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest
from conftest import GitRepoFactory

from codebase_qa.chunking import whitespace_token_count
from codebase_qa.embeddings import Embedder, HashEmbedder
from codebase_qa.indexer import index_repo
from codebase_qa.retrieval import Filters, RetrievalConfig, Retriever, SearchResult

FILES = {
    "net/proxy.py": "def proxy_from_env(environ):\n    return environ.get('HTTPS_PROXY')\n",
    "net/retry.py": "def retry_request(request, attempts):\n    return send request again\n",
    "tests/test_proxy.py": "def test_proxy_from_env():\n    assert proxy_from_env({})\n",
    # Filler keeps FTS5's IDF positive for the terms above.
    **{f"net/util_{i}.py": f"def helper_{i}(value):\n    return value + {i}\n" for i in range(8)},
}


class KeywordReranker:
    """Scores a text by how often it mentions ``keyword``."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.calls: list[int] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append(len(texts))
        return [float(text.count(self.keyword)) for text in texts]


def unavailable() -> NoReturn:
    raise AssertionError("model should not be loaded")


@pytest.fixture
def index_dir(make_repo: GitRepoFactory, tmp_path: Path) -> Path:
    repo = make_repo(FILES)
    index_repo(repo, tmp_path / "index", HashEmbedder(), whitespace_token_count)
    return tmp_path / "index"


def symbols(results: list[SearchResult]) -> list[str]:
    return [r.chunk.symbol_path for r in results]


def test_hybrid_records_each_retrievers_rank_and_the_rrf_score(index_dir: Path) -> None:
    with Retriever.open(index_dir, embedder=HashEmbedder) as retriever:
        results = retriever.retrieve("proxy_from_env", RetrievalConfig(mode="hybrid", k=20))

    assert "proxy_from_env" in symbols(results)
    for rank, r in enumerate(results, 1):
        assert r.ranks["hybrid"] == rank
        expected = sum(1 / (60 + r.ranks[stage]) for stage in ("dense", "bm25") if stage in r.ranks)
        assert r.score == pytest.approx(expected)
        assert r.scores["rrf"] == r.score
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_hybrid_keeps_chunks_only_one_retriever_found(index_dir: Path) -> None:
    with Retriever.open(index_dir, embedder=HashEmbedder) as retriever:
        results = retriever.retrieve("proxy_from_env", RetrievalConfig(mode="hybrid", k=20))

    dense_only = [r for r in results if "bm25" not in r.ranks]
    assert dense_only
    assert all("dense" in r.ranks for r in dense_only)


def test_hybrid_applies_filters_to_both_retrievers(index_dir: Path) -> None:
    config = RetrievalConfig(mode="hybrid", k=20, filters=Filters(roles=("test",)))
    with Retriever.open(index_dir, embedder=HashEmbedder) as retriever:
        results = retriever.retrieve("proxy_from_env", config)

    assert symbols(results) == ["test_proxy_from_env"]
    assert set(results[0].ranks) == {"dense", "bm25", "hybrid"}


def test_rerank_reorders_the_head_and_records_movement(index_dir: Path) -> None:
    reranker = KeywordReranker("request")
    query = "proxy environ request"
    config = RetrievalConfig(mode="dense", k=6, rerank=True, rerank_k=3)
    with Retriever.open(index_dir, embedder=HashEmbedder, reranker=lambda: reranker) as retriever:
        before = retriever.retrieve(query, RetrievalConfig(mode="dense", k=6))
        after = retriever.retrieve(query, config)

    first_stage = symbols(before)
    assert "retry_request" in first_stage[:3]
    assert first_stage[0] != "retry_request"
    assert reranker.calls == [3]
    assert symbols(after)[0] == "retry_request"
    assert after[0].ranks == {"dense": first_stage.index("retry_request") + 1, "rerank": 1}
    assert (
        after[0].score
        == after[0].scores["rerank"]
        == after[0].chunk.embedding_text.count("request")
    )
    assert sorted(symbols(after[:3])) == sorted(first_stage[:3])
    assert symbols(after[3:]) == first_stage[3:]
    assert all("rerank" not in r.ranks for r in after[3:])


def test_rerank_depth_reaches_rerank_k_even_when_k_is_smaller(index_dir: Path) -> None:
    reranker = KeywordReranker("request")
    config = RetrievalConfig(mode="dense", k=1, rerank=True, rerank_k=4)
    with Retriever.open(index_dir, embedder=HashEmbedder, reranker=lambda: reranker) as retriever:
        results = retriever.retrieve("environment proxy", config)

    assert reranker.calls == [4]
    assert len(results) == 1


def test_models_load_lazily_and_only_once(index_dir: Path) -> None:
    loads: list[str] = []

    def embedder() -> Embedder:
        loads.append("embedder")
        return HashEmbedder()

    with Retriever.open(index_dir, embedder=embedder, reranker=unavailable) as retriever:
        retriever.retrieve("proxy", RetrievalConfig(mode="bm25"))
        assert loads == []
        retriever.retrieve("proxy", RetrievalConfig(mode="dense"))
        retriever.retrieve("proxy", RetrievalConfig(mode="hybrid"))

    assert loads == ["embedder"]


def test_rerank_without_a_reranker_is_an_error(index_dir: Path) -> None:
    with Retriever.open(index_dir) as retriever, pytest.raises(ValueError, match="reranker"):
        retriever.retrieve("proxy", RetrievalConfig(mode="bm25", rerank=True))


@pytest.mark.parametrize(
    "overrides", [{"k": 0}, {"candidate_k": 0}, {"rerank_k": 0}, {"rrf_k": -1}]
)
def test_config_rejects_out_of_range_values(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(**overrides)  # pyright: ignore[reportArgumentType]
