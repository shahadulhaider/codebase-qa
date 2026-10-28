import pytest

from codebase_qa.chunking import Chunk, SymbolKind
from codebase_qa.eval.metrics import (
    GoldTarget,
    first_hit_ranks,
    level_targets,
    recall_at,
    reciprocal_rank,
)


def ts(path: str, symbol: str, start: int = 1, end: int = 5, kind: SymbolKind = "method") -> Chunk:
    return Chunk(
        file_path=path,
        language="python",
        text=f"{symbol} {start}",
        start_line=start,
        end_line=end,
        symbol_kind=kind if symbol else "module",
        chunker="treesitter",
        symbol_name=symbol.rsplit(".", 1)[-1],
        symbol_path=symbol,
    )


def win(path: str, start: int, end: int) -> Chunk:
    return Chunk(path, "python", f"lines {start}", start, end, "window", "window")


CLIENT_SEND = GoldTarget("client.py", "Client.send")
CONFIG = GoldTarget("config.py", "Timeout")
SPANS = {CLIENT_SEND: ((40, 60),), CONFIG: ((10, 30),)}


def test_split_pieces_of_one_symbol_share_a_rank() -> None:
    results = [
        ts("client.py", "Client.request", 1, 20),
        ts("client.py", "Client.request", 18, 39),
        ts("client.py", "Client.send", 40, 60),
    ]
    assert first_hit_ranks(results, [CLIENT_SEND], SPANS, "symbol") == {CLIENT_SEND: 2}


def test_file_level_dedupes_by_file() -> None:
    results = [
        ts("other.py", "a"),
        ts("other.py", "b"),
        ts("client.py", "Client.request"),
        ts("config.py", "Timeout"),
    ]
    ranks = first_hit_ranks(results, [CLIENT_SEND, CONFIG], SPANS, "file")
    assert ranks == {GoldTarget("client.py"): 2, GoldTarget("config.py"): 3}


def test_method_hits_its_class_but_not_a_prefix_sibling() -> None:
    gold = GoldTarget("client.py", "Client")
    results = [ts("client.py", "ClientState"), ts("client.py", "Client.send")]
    assert first_hit_ranks(results, [gold], {gold: ((1, 99),)}, "symbol") == {gold: 2}


def test_window_hits_every_overlapping_gold_symbol_at_its_own_rank() -> None:
    a = GoldTarget("m.py", "a")
    b = GoldTarget("m.py", "b")
    spans = {a: ((1, 10),), b: ((11, 20),)}
    results = [win("m.py", 30, 40), win("m.py", 8, 14)]
    assert first_hit_ranks(results, [a, b], spans, "symbol") == {a: 2, b: 2}


def test_window_touching_a_span_boundary_counts_and_disjoint_does_not() -> None:
    results = [win("client.py", 61, 80), win("client.py", 20, 40)]
    assert first_hit_ranks(results, [CLIENT_SEND], SPANS, "symbol") == {CLIENT_SEND: 2}


def test_window_between_two_declarations_of_one_name_misses() -> None:
    gold = GoldTarget("core.ts", "make")
    spans = {gold: ((8, 12), (51, 145))}
    results = [win("core.ts", 20, 40), win("core.ts", 140, 160)]
    assert first_hit_ranks(results, [gold], spans, "symbol") == {gold: 2}


def test_treesitter_module_chunk_never_hits_a_symbol() -> None:
    results = [ts("client.py", "", 40, 60, kind="module")]
    assert first_hit_ranks(results, [CLIENT_SEND], SPANS, "symbol") == {}


def test_file_only_gold_is_hit_by_any_chunk_of_the_file() -> None:
    readme = GoldTarget("README.md")
    results = [ts("client.py", "x"), win("README.md", 5, 9)]
    assert first_hit_ranks(results, [readme], {}, "symbol") == {readme: 2}


def test_level_targets() -> None:
    gold = [CLIENT_SEND, GoldTarget("client.py", "Client.request"), CONFIG]
    assert level_targets(gold, "file") == [GoldTarget("client.py"), GoldTarget("config.py")]
    assert len(level_targets(gold, "symbol")) == 3


def test_recall_and_reciprocal_rank_by_hand() -> None:
    ranks = {CLIENT_SEND: 3, CONFIG: 7}
    assert recall_at(ranks, 2, 1) == 0.0
    assert recall_at(ranks, 2, 5) == 0.5
    assert recall_at(ranks, 2, 10) == 1.0
    assert recall_at(ranks, 3, 10) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranks, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranks, 2) == 0.0
    assert reciprocal_rank({}, 10) == 0.0


def test_recall_needs_gold() -> None:
    with pytest.raises(ValueError, match="gold"):
        recall_at({}, 0, 5)
