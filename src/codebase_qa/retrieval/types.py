"""Retrieval request and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from codebase_qa.chunking import Chunk

Mode = Literal["dense", "bm25"]


@dataclass(frozen=True)
class Filters:
    """Hard constraints applied inside every retriever, never after top-k.

    ``path_glob`` uses SQLite GLOB semantics: case-sensitive, and ``*``
    matches across ``/``.
    """

    languages: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    """File roles: source, test, docs or config (see ``role_for_path``)."""
    path_glob: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    """Higher is better: cosine similarity for dense, negated FTS5 bm25 for BM25."""
    scores: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    """Per-retriever raw values, e.g. ``{"dense_distance": 0.41}``."""

    @property
    def location(self) -> str:
        return f"{self.chunk.file_path}:{self.chunk.start_line}-{self.chunk.end_line}"
