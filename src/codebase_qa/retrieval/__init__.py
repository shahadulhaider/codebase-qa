"""Query an index: dense, BM25, metadata filters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from codebase_qa.chunking import ChunkerName
from codebase_qa.embeddings import Embedder
from codebase_qa.indexer import CATALOG_FILE, VECTORS_DIR
from codebase_qa.retrieval.dense import dense_search
from codebase_qa.retrieval.filters import ResolvedFilters
from codebase_qa.retrieval.lexical import bm25_search, fts_query
from codebase_qa.retrieval.types import Filters, Mode, SearchResult
from codebase_qa.store import Build, Store
from codebase_qa.vector_index import ChromaVectorIndex

__all__ = [
    "Filters",
    "IndexNotFoundError",
    "Mode",
    "SearchResult",
    "Searcher",
    "StaleIndexError",
    "fts_query",
]


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
    ) -> list[SearchResult]:
        """Ranked results for ``query``.

        ``embedder`` is a factory, required for dense mode, so a model is only
        loaded when a query actually needs one.
        """
        resolved = ResolvedFilters.resolve(self._store, filters or Filters())
        if mode == "bm25":
            ranked = bm25_search(self._store, self.build.chunker, query, k, resolved)
            return self._results(ranked, lambda score: score, "bm25")
        if embedder is None:
            raise ValueError("dense search needs an embedder")
        ranked = dense_search(self._vectors(), embedder(), query, k, resolved)
        return self._results(ranked, lambda distance: 1.0 - distance, "dense_distance")

    def _vectors(self) -> ChromaVectorIndex:
        if self._collection is None:
            self._collection = ChromaVectorIndex.open(
                self._index_dir / VECTORS_DIR, self.build.collection
            )
        return self._collection

    def _results(
        self, ranked: list[tuple[str, float]], to_score: Callable[[float], float], raw: str
    ) -> list[SearchResult]:
        chunks = self._store.chunks_by_id(self.build.chunker, [chunk_id for chunk_id, _ in ranked])
        return [
            SearchResult(chunks[chunk_id], to_score(value), {raw: value})
            for chunk_id, value in ranked
            if chunk_id in chunks
        ]
