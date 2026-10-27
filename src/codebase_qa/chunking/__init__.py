"""Splitting source files into retrievable, citable chunks."""

from __future__ import annotations

from codebase_qa.chunking.base import (
    Chunk,
    Chunker,
    ChunkerName,
    SymbolKind,
    TokenCounter,
    whitespace_token_count,
)
from codebase_qa.chunking.languages import LanguageRegistry, LanguageSpec
from codebase_qa.chunking.selection import FileRoute, route_file
from codebase_qa.chunking.treesitter import TreeSitterChunker
from codebase_qa.chunking.window import LineWindowChunker

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkerName",
    "FileRoute",
    "LanguageRegistry",
    "LanguageSpec",
    "LineWindowChunker",
    "SymbolKind",
    "TokenCounter",
    "TreeSitterChunker",
    "chunk_file",
    "route_file",
    "whitespace_token_count",
]


def chunk_file(
    file_path: str,
    source: str,
    route: FileRoute,
    *,
    chunker: ChunkerName,
    token_count: TokenCounter,
    max_tokens: int = 256,
    registry: LanguageRegistry | None = None,
) -> list[Chunk]:
    """Chunk one file.

    With the ``treesitter`` chunker, code files are split by definition and
    other files fall back to line windows; the ``window`` chunker windows
    everything, which makes it a like-for-like baseline.
    """
    if chunker == "treesitter" and route.syntax_aware:
        return TreeSitterChunker(token_count, registry=registry, max_tokens=max_tokens).chunk(
            file_path, route.language, source
        )
    return LineWindowChunker(token_count, max_tokens=max_tokens).chunk(
        file_path, route.language, source
    )
