"""Reciprocal Rank Fusion of several ranked id lists."""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_RRF_K = 60


def rrf(lists: Sequence[Sequence[str]], k: int = DEFAULT_RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked lists into (id, score) pairs, best first.

    Each list contributes ``1 / (k + rank)`` for every id it contains, with
    1-based ranks. Only ranks are used, so retrievers whose scores live on
    unrelated scales can be combined without weights. Ties are broken by the
    best rank in any single list, then by list order, so output is stable.
    """
    if k < 0:
        raise ValueError("rrf k must be non-negative")
    scores: dict[str, float] = {}
    best: dict[str, tuple[int, int]] = {}
    for list_index, ids in enumerate(lists):
        for rank, item in enumerate(dict.fromkeys(ids), 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best[item] = min(best.get(item, (rank, list_index)), (rank, list_index))
    order = sorted(scores, key=lambda item: (-scores[item], best[item]))
    return [(item, scores[item]) for item in order]
