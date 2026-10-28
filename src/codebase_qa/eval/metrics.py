"""Rank-based retrieval metrics over chunker-independent gold targets.

Gold is a file, or a symbol within a file, never a chunk id, so one gold set
scores every chunker. Retrieved chunks are mapped to targets and duplicates
are dropped before ranks are counted: two pieces of one split function occupy
one rank, not two.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from codebase_qa.chunking import Chunk

Level = Literal["file", "symbol"]
LEVELS: tuple[Level, ...] = ("file", "symbol")


@dataclass(frozen=True)
class GoldTarget:
    file: str
    symbol: str = ""
    """A treesitter ``symbol_path``; empty when the whole file is the target."""

    def __str__(self) -> str:
        return f"{self.file}::{self.symbol}" if self.symbol else self.file


Span = tuple[int, int]
"""1-based inclusive line range."""

Spans = tuple[Span, ...]
"""Where a gold symbol is defined at the pinned commit. TypeScript can declare one
name more than once (an interface and a const), so a symbol may have several."""


def level_targets(gold: Sequence[GoldTarget], level: Level) -> list[GoldTarget]:
    """The distinct targets scored at ``level``; file level keeps only the files."""
    if level == "file":
        return list(dict.fromkeys(GoldTarget(g.file) for g in gold))
    return list(dict.fromkeys(gold))


def first_hit_ranks(
    results: Sequence[Chunk],
    gold: Sequence[GoldTarget],
    spans: Mapping[GoldTarget, Spans],
    level: Level,
) -> dict[GoldTarget, int]:
    """1-based rank at which each gold target is first hit; unhit targets are absent.

    At symbol level a chunk hits a gold symbol when its ``symbol_path`` is that
    symbol or a member of it (a method hits its class). Window chunks carry no
    symbol, so they hit every gold symbol whose spans their lines overlap, all at
    the window's own rank. A file-only gold target is hit by any chunk of that
    file at either level.
    """
    targets = level_targets(gold, level)
    ranks: dict[GoldTarget, int] = {}
    seen: set[tuple[str | int, ...]] = set()
    rank = 0
    for chunk in results:
        key = _result_key(chunk, level)
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        for target in targets:
            if target not in ranks and _hits(chunk, target, spans):
                ranks[target] = rank
    return ranks


def recall_at(ranks: Mapping[GoldTarget, int], n_gold: int, k: int) -> float:
    if n_gold == 0:
        raise ValueError("a question needs at least one gold target")
    return sum(1 for r in ranks.values() if r <= k) / n_gold


def reciprocal_rank(ranks: Mapping[GoldTarget, int], cutoff: int) -> float:
    """1/rank of the first gold hit within ``cutoff``, else 0."""
    first = min(ranks.values(), default=None)
    return 0.0 if first is None or first > cutoff else 1.0 / first


def _result_key(chunk: Chunk, level: Level) -> tuple[str | int, ...]:
    if level == "file":
        return (chunk.file_path,)
    if chunk.symbol_path:
        return (chunk.file_path, chunk.symbol_path)
    return (chunk.file_path, chunk.start_line, chunk.end_line)


def _hits(chunk: Chunk, target: GoldTarget, spans: Mapping[GoldTarget, Spans]) -> bool:
    if chunk.file_path != target.file:
        return False
    if not target.symbol:
        return True
    if chunk.chunker == "window":
        return any(
            chunk.start_line <= end and start <= chunk.end_line for start, end in spans[target]
        )
    return chunk.symbol_path == target.symbol or chunk.symbol_path.startswith(target.symbol + ".")
