"""Vector retrieval over a build's Chroma collection."""

from __future__ import annotations

from codebase_qa.embeddings import Embedder
from codebase_qa.retrieval.filters import ResolvedFilters
from codebase_qa.vector_index import ChromaVectorIndex

OVERSAMPLE = 4


def dense_search(
    index: ChromaVectorIndex, embedder: Embedder, query: str, k: int, filters: ResolvedFilters
) -> list[tuple[str, float]]:
    """Top ``k`` (chunk id, cosine distance) pairs, nearest first."""
    if filters.empty:
        return []
    vector = embedder.embed_query(query)
    where = filters.chroma_where()
    if filters.files is None or filters.pushes_files_to_vector_store:
        return [(h.id, h.distance) for h in index.query(vector, k, where)]
    # Too many files for an $in clause: oversample and post-filter. This can
    # return fewer than k results for very selective globs.
    hits = index.query(vector, k * OVERSAMPLE, where)
    kept = [(h.id, h.distance) for h in hits if filters.accepts_file(str(h.metadata["file_path"]))]
    return kept[:k]
