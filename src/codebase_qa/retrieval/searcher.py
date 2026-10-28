"""Query one build of an index: dense, BM25 and hybrid fusion."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from codebase_qa.chunking import ChunkerName
from codebase_qa.embeddings import Embedder
from codebase_qa.indexer import CATALOG_FILE, VECTORS_DIR
from codebase_qa.retrieval.dense import dense_search
from codebase_qa.retrieval.filters import ResolvedFilters
from codebase_qa.retrieval.fusion import DEFAULT_RRF_K, rrf
from codebase_qa.retrieval.lexical import bm25_search
from codebase_qa.retrieval.types import Filters, Mode, SearchResult
from codebase_qa.store import Build, Store
from codebase_qa.vector_index import ChromaVectorIndex

DEFAULT_CANDIDATE_K = 50


class IndexNotFoundError(LookupError):
    pass


class StaleIndexError(RuntimeError):
    pass


class Searcher:
    """Runs queries against one build of an index.

    The build defaults to the most recently indexed one; ``model_id`` and
    ``chunker`` narrow the choice. A build that is not at the catalog's current
    revision is refused, because its vectors no longer match the catalog.
    """

    def __init__(self, index_dir: Path, store: Store, build: Build) -> None:
        self._index_dir = index_dir
        self._store = store
        self.build = build
        self._collection: ChromaVectorIndex | None = None

    @classmethod
    def open(
        cls,
        index_dir: Path,
        *,
        model_id: str | None = None,
        chunker: ChunkerName | None = None,
    ) -> Searcher:
        if not (index_dir / CATALOG_FILE).exists():
            raise IndexNotFoundError(f"no index at {index_dir}; run `codebase-qa index` first")
        store = Store.open(index_dir / CATALOG_FILE)
        builds = [
            b
            for b in store.builds()
            if (model_id is None or b.model_id == model_id)
            and (chunker is None or b.chunker == chunker)
        ]
        if not builds:
            store.close()
            raise IndexNotFoundError(f"no matching build in {index_dir}")
        build = builds[0]
        current = store.current_rev or ""
        if build.indexed_rev != current:
            store.close()
            raise StaleIndexError(
                f"{build.collection} was indexed at {build.indexed_rev[:12]} but the index is at "
                f"{current[:12]}; re-run `codebase-qa index --chunker "
                f"{build.chunker} --model {build.model_id}`"
            )
        return cls(index_dir, store, build)

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> Searcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(
        self,
        query: str,
        *,
        mode: Mode,
        k: int = 10,
        filters: Filters | None = None,
        embedder: Callable[[], Embedder] | None = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[SearchResult]:
        """Ranked results for ``query``.

        ``embedder`` is a factory, required for dense and hybrid modes, so a
        model is only loaded when a query actually needs one. Hybrid mode
        fetches ``candidate_k`` results from each retriever and fuses them
        with RRF before keeping the top ``k``.
        """
        resolved = ResolvedFilters.resolve(self._store, filters or Filters())
        if mode == "bm25":
            ranked = bm25_search(self._store, self.build.chunker, query, k, resolved)
            return self._results([(i, s, {"bm25": s}, {"bm25": r}) for i, s, r in _ranks(ranked)])
        if embedder is None:
            raise ValueError(f"{mode} search needs an embedder")
        if mode == "dense":
            ranked = dense_search(self._vectors(), embedder(), query, k, resolved)
            return self._results(
                [(i, 1.0 - d, {"dense_distance": d}, {"dense": r}) for i, d, r in _ranks(ranked)]
            )
        depth = max(k, candidate_k)
        dense = dense_search(self._vectors(), embedder(), query, depth, resolved)
        lexical = bm25_search(self._store, self.build.chunker, query, depth, resolved)
        fused = rrf([[i for i, _ in dense], [i for i, _ in lexical]], rrf_k)[:k]
        stages = {
            ("dense", "dense_distance"): {i: (r, d) for i, d, r in _ranks(dense)},
            ("bm25", "bm25"): {i: (r, s) for i, s, r in _ranks(lexical)},
        }
        rows: list[_Row] = []
        for rank, (chunk_id, score) in enumerate(fused, 1):
            scores, ranks = {"rrf": score}, {"hybrid": rank}
            for (stage, raw), positions in stages.items():
                if chunk_id in positions:
                    ranks[stage], scores[raw] = positions[chunk_id]
            rows.append((chunk_id, score, scores, ranks))
        return self._results(rows)

    def _vectors(self) -> ChromaVectorIndex:
        if self._collection is None:
            self._collection = ChromaVectorIndex.open(
                self._index_dir / VECTORS_DIR, self.build.collection
            )
        return self._collection

    def _results(self, rows: list[_Row]) -> list[SearchResult]:
        chunks = self._store.chunks_by_id(self.build.chunker, [row[0] for row in rows])
        return [
            SearchResult(chunks[chunk_id], score, scores, ranks)
            for chunk_id, score, scores, ranks in rows
            if chunk_id in chunks
        ]


_Row = tuple[str, float, dict[str, float], dict[str, int]]


def _ranks(ranked: list[tuple[str, float]]) -> list[tuple[str, float, int]]:
    return [(chunk_id, value, rank) for rank, (chunk_id, value) in enumerate(ranked, 1)]
