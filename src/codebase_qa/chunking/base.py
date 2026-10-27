"""Chunk model shared by all chunkers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

SymbolKind = Literal["module", "class", "function", "method", "interface", "type", "enum", "window"]
ChunkerName = Literal["treesitter", "window"]

TokenCounter = Callable[[str], int]
"""Counts tokens in a string, including any special tokens the model adds."""


def whitespace_token_count(text: str) -> int:
    """Model-free counter for tests and offline use."""
    return len(text.split())


@dataclass(frozen=True)
class Chunk:
    """A contiguous, line-aligned span of one file.

    Line numbers are 1-based and inclusive. ``symbol_name`` and ``symbol_path``
    are empty strings for module-level and window chunks, because the vector
    store does not accept null metadata.
    """

    file_path: str
    language: str
    text: str
    start_line: int
    end_line: int
    symbol_kind: SymbolKind
    chunker: ChunkerName
    symbol_name: str = ""
    symbol_path: str = ""

    @property
    def header(self) -> str:
        """Location context prepended to the text for dense embedding.

        A bare ``def get(self):`` is ambiguous; the path and qualified symbol
        tell the embedder what it belongs to.
        """
        if self.symbol_path:
            return f"# {self.file_path}\n# {self.symbol_path} ({self.symbol_kind})\n"
        return f"# {self.file_path} ({self.symbol_kind})\n"

    @property
    def embedding_text(self) -> str:
        return self.header + self.text

    @property
    def content_sha(self) -> str:
        """Identifies the embedded input; equal values can share one vector."""
        return hashlib.sha256(self.embedding_text.encode()).hexdigest()

    @property
    def chunk_id(self) -> str:
        """Stable identity of this chunk within a file and chunker."""
        key = "\x1f".join(
            (self.file_path, self.symbol_path, str(self.start_line), self.chunker, self.text)
        )
        return hashlib.sha256(key.encode()).hexdigest()


class Chunker(Protocol):
    name: ChunkerName

    def chunk(self, file_path: str, language: str, source: str) -> list[Chunk]: ...


@dataclass(frozen=True)
class Line:
    number: int
    text: str


def split_to_budget(
    lines: Sequence[Line],
    *,
    header: str,
    token_count: TokenCounter,
    max_tokens: int,
    overlap_lines: int,
) -> list[list[Line]]:
    """Greedily pack lines into windows whose header + body fit ``max_tokens``.

    Per-line costs are counted once and summed, which matches whole-text
    counts for tokenizers that split on whitespace boundaries. Consecutive
    windows share up to ``overlap_lines`` trailing lines. A line that exceeds
    the budget on its own (e.g. a paragraph written on one line) is split at
    word boundaries; every piece keeps that line's number.
    """
    if not lines:
        return []
    overhead = token_count("")
    budget = max(1, max_tokens - token_count(header))
    lines = [piece for line in lines for piece in _split_line(line, budget, token_count, overhead)]
    costs = [token_count(line.text) - overhead for line in lines]

    windows: list[list[Line]] = []
    start = 0
    while start < len(lines):
        end = start
        used = 0
        while end < len(lines) and (end == start or used + costs[end] <= budget):
            used += costs[end]
            end += 1
        windows.append(list(lines[start:end]))
        if end >= len(lines):
            break
        start = max(end - overlap_lines, start + 1)
    return windows


def _split_line(line: Line, budget: int, token_count: TokenCounter, overhead: int) -> list[Line]:
    if token_count(line.text) - overhead <= budget:
        return [line]
    pieces: list[Line] = []
    words: list[str] = []
    used = 0
    for word in line.text.split(" "):
        cost = token_count(word) - overhead
        if words and used + cost > budget:
            pieces.append(Line(line.number, " ".join(words)))
            words, used = [], 0
        words.append(word)
        used += cost
    if words:
        pieces.append(Line(line.number, " ".join(words)))
    return pieces
