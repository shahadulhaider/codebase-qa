import pytest

from codebase_qa.retrieval import rrf


def test_scores_are_summed_reciprocal_ranks() -> None:
    fused = dict(rrf([["a", "b", "c"], ["c", "a"]], k=60))

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_agreement_beats_a_single_top_rank() -> None:
    dense = ["solo", *(f"d{i}" for i in range(6)), "both"]
    bm25 = [*(f"b{i}" for i in range(7)), "both"]

    fused = [item for item, _ in rrf([dense, bm25], k=60)]

    assert fused[0] == "both"
    assert fused.index("both") < fused.index("solo")


def test_small_k_favours_the_best_single_rank() -> None:
    dense = ["solo", *(f"d{i}" for i in range(6)), "both"]
    bm25 = [*(f"b{i}" for i in range(7)), "both"]

    fused = [item for item, _ in rrf([dense, bm25], k=0)]

    assert fused[0] == "solo"


def test_items_in_only_one_list_are_kept() -> None:
    fused = [item for item, _ in rrf([["a"], ["b"], []])]

    assert sorted(fused) == ["a", "b"]


def test_ties_break_by_best_rank_then_list_order() -> None:
    # Each id appears in one list only, so both rank-1 ids tie, as do both
    # rank-2 ids; the first list wins each tie.
    fused = rrf([["b1", "a1"], ["a2", "b2"]], k=60)

    assert [item for item, _ in fused] == ["b1", "a2", "a1", "b2"]
    assert fused[0][1] == fused[1][1]


def test_duplicates_within_a_list_count_once_at_their_best_rank() -> None:
    fused = dict(rrf([["a", "a", "b"]], k=60))

    assert fused == {"a": pytest.approx(1 / 61), "b": pytest.approx(1 / 62)}


def test_negative_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rrf([["a"]], k=-1)
