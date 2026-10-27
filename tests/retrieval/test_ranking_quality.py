from pathlib import Path

import pytest
from conftest import GitRepoFactory

from codebase_qa import store
from codebase_qa.chunking import whitespace_token_count
from codebase_qa.embeddings import HashEmbedder
from codebase_qa.indexer import index_repo
from codebase_qa.retrieval import Searcher

# The definition mentions its own name once; a caller and a test mention it
# repeatedly, so plain term frequency ranks them above the definition.
FILES = {
    "shop/pricing.py": (
        "def compute_total_price(items):\n"
        "    subtotal = sum(item.price * item.quantity for item in items)\n"
        "    return round(subtotal * 1.2, 2)\n"
    ),
    "shop/checkout.py": (
        "def checkout(cart):\n"
        "    total = compute_total_price(cart.items)\n"
        "    preview = compute_total_price(cart.items[:1])\n"
        "    log(compute_total_price, total, preview)\n"
        "    return total\n"
    ),
    "tests/test_pricing.py": (
        "def test_compute_total_price():\n"
        "    assert compute_total_price([]) == 0\n"
        "    assert compute_total_price([item(1, 2)]) == 2.4\n"
        "    assert compute_total_price([item(1, 1)]) == 1.2\n"
    ),
    # Unrelated files: with too few documents, FTS5's IDF for terms present in
    # half or more of them is clamped to ~0 and every score collapses.
    **{
        f"shop/module_{i}.py": f"def helper_{i}(value):\n    return value + {i}\n" for i in range(8)
    },
}


@pytest.fixture
def index_dir(make_repo: GitRepoFactory, tmp_path: Path) -> Path:
    index_repo(make_repo(FILES), tmp_path / "index", HashEmbedder(), whitespace_token_count)
    return tmp_path / "index"


def top_symbols(index_dir: Path) -> list[str]:
    with Searcher.open(index_dir) as searcher:
        results = searcher.search("compute_total_price", mode="bm25", k=3)
    return [r.chunk.symbol_path for r in results]


def test_bm25_ranks_the_definition_above_callers_and_tests(index_dir: Path) -> None:
    assert top_symbols(index_dir)[0] == "compute_total_price"


def test_without_the_symbol_weight_callers_win(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "_BM25_WEIGHTS", "1.0, 0.5, 0.0")

    assert top_symbols(index_dir)[0] != "compute_total_price"
