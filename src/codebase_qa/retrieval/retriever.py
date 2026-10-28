"""Configured retrieval: first-stage search plus optional reranking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from codebase_qa.chunking import ChunkerName
from codebase_qa.embeddings import Embedder
from codebase_qa.retrieval.fusion import DEFAULT_RRF_K
from codebase_qa.retrieval.rerank import Reranker
from codebase_qa.retrieval.searcher import DEFAULT_CANDIDATE_K, Searcher
from codebase_qa.retrieval.types import Filters, Mode, SearchResult


@dataclass(frozen=True)
class RetrievalConfig:
    mode: Mode = "dense"
    k: int = 10
    candidate_k: int = DEFAULT_CANDIDATE_K
    """Results fetched from each retriever before hybrid fusion."""
    rrf_k: int = DEFAULT_RRF_K
    rerank: bool = False
    rerank_k: int = 20
    """Top first-stage results the reranker reorders; any beyond keep their order."""
    filters: Filters = field(default_factory=Filters)

    def __post_init__(self) -> None:
        for name in ("k", "candidate_k", "rerank_k"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")


class Retriever:
    """Runs a :class:`RetrievalConfig` against one build of an index.

    The embedder and reranker are factories, called at most once on first
    use and then reused, so repeated queries pay the model load only once and
    modes that do not need a model never load it.
    """

    def __init__(
        self,
        searcher: Searcher,
        *,
        embedder: Callable[[], Embedder] | None = None,
        reranker: Callable[[], Reranker] | None = None,
    ) -> None:
        self.searcher = searcher
        self._embedder_factory = embedder
        self._reranker_factory = reranker
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None

    @classmethod
    def open(
        cls,
        index_dir: Path,
        *,
        model_id: str | None = None,
        chunker: ChunkerName | None = None,
        embedder: Callable[[], Embedder] | None = None,
        reranker: Callable[[], Reranker] | None = None,
    ) -> Retriever:
        searcher = Searcher.open(index_dir, model_id=model_id, chunker=chunker)
        return cls(searcher, embedder=embedder, reranker=reranker)

    def close(self) -> None:
        self.searcher.close()

    def __enter__(self) -> Retriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def retrieve(self, query: str, config: RetrievalConfig | None = None) -> list[SearchResult]:
        config = config or RetrievalConfig()
        depth = max(config.k, config.rerank_k) if config.rerank else config.k
        results = self.searcher.search(
            query,
            mode=config.mode,
            k=depth,
            filters=config.filters,
            embedder=self._load_embedder if self._embedder_factory else None,
            candidate_k=config.candidate_k,
            rrf_k=config.rrf_k,
        )
        if config.rerank and results:
            results = self._rerank(query, results, config.rerank_k)
        return results[: config.k]

    def _rerank(self, query: str, results: list[SearchResult], rerank_k: int) -> list[SearchResult]:
        head, tail = results[:rerank_k], results[rerank_k:]
        scores = self._load_reranker().score(query, [r.chunk.embedding_text for r in head])
        order = sorted(range(len(head)), key=lambda i: -scores[i])
        reranked = [
            replace(
                head[i],
                score=scores[i],
                scores={**head[i].scores, "rerank": scores[i]},
                ranks={**head[i].ranks, "rerank": rank},
            )
            for rank, i in enumerate(order, 1)
        ]
        return reranked + tail

    def _load_embedder(self) -> Embedder:
        if self._embedder is None:
            assert self._embedder_factory is not None
            self._embedder = self._embedder_factory()
        return self._embedder

    def _load_reranker(self) -> Reranker:
        if self._reranker is None:
            if self._reranker_factory is None:
                raise ValueError("reranking needs a reranker")
            self._reranker = self._reranker_factory()
        return self._reranker
