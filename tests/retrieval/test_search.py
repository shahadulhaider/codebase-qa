from pathlib import Path

import pytest
from conftest import GitRepoFactory, commit_files

from codebase_qa.chunking import ChunkerName, whitespace_token_count
from codebase_qa.embeddings import HashEmbedder
from codebase_qa.indexer import IndexOptions, index_repo
from codebase_qa.retrieval import (
    Filters,
    IndexNotFoundError,
    Searcher,
    SearchResult,
    StaleIndexError,
    fts_query,
)
from codebase_qa.retrieval import filters as filters_module

FILES = {
    "src/http/client.py": (
        "class Client:\n"
        "    token = None\n\n"
        "    def send(self, request):\n"
        "        return transport token request\n\n"
        "    def close(self):\n"
        "        return token\n"
    ),
    "src/http/redirects.py": (
        "def follow_redirects(response):\n    return token response location\n"
    ),
    "src/config.py": "def load_config(path):\n    return token path settings\n",
    "web/users.ts": (
        "export function getUserById(id: number) {\n  return token;\n}\n\n"
        "export interface User {\n  token: string;\n}\n"
    ),
}


def build(repo: Path, index_dir: Path, chunker: ChunkerName = "treesitter") -> None:
    index_repo(
        repo, index_dir, HashEmbedder(), whitespace_token_count, IndexOptions(chunker=chunker)
    )


@pytest.fixture
def index_dir(make_repo: GitRepoFactory, tmp_path: Path) -> Path:
    repo = make_repo(FILES)
    build(repo, tmp_path / "index")
    return tmp_path / "index"


def symbols(results: list[SearchResult]) -> list[str]:
    return [r.chunk.symbol_path for r in results]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("self-hosted", '"self" OR "hosted"'),
        ('say "hi" now', '"say" OR "hi" OR "now"'),
        ("getUserById", '"getuserbyid" OR "get" OR "user" OR "by" OR "id"'),
        ("a:b NEAR(c)", '"a" OR "b" OR "near" OR "c"'),
        ("?!-", None),
    ],
)
def test_fts_query_quotes_every_term(text: str, expected: str | None) -> None:
    assert fts_query(text) == expected


def test_bm25_ranks_exact_identifier_first(index_dir: Path) -> None:
    with Searcher.open(index_dir) as searcher:
        results = searcher.search("getUserById", mode="bm25", k=3)

    assert symbols(results)[0] == "getUserById"
    assert results[0].score > 0
    assert results[0].location == "web/users.ts:1-3"


def test_bm25_matches_camel_case_parts(index_dir: Path) -> None:
    with Searcher.open(index_dir) as searcher:
        results = searcher.search("user", mode="bm25", k=5)

    assert "getUserById" in symbols(results)


def test_dense_scores_are_cosine_similarities(index_dir: Path) -> None:
    with Searcher.open(index_dir) as searcher:
        results = searcher.search("load config settings", mode="dense", k=3, embedder=HashEmbedder)

    assert symbols(results)[0] == "load_config"
    distance = results[0].scores["dense_distance"]
    assert results[0].score == pytest.approx(1 - distance)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_dense_requires_an_embedder(index_dir: Path) -> None:
    with Searcher.open(index_dir) as searcher, pytest.raises(ValueError, match="embedder"):
        searcher.search("x", mode="dense")


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (Filters(languages=("typescript",)), {"getUserById", "User"}),
        (Filters(kinds=("method",)), {"Client.send", "Client.close"}),
        (
            Filters(path_glob="src/http/*"),
            {"Client", "Client.send", "Client.close", "follow_redirects"},
        ),
        (Filters(languages=("python",), kinds=("function",)), {"follow_redirects", "load_config"}),
    ],
)
def test_filters_give_both_retrievers_the_same_candidates(
    index_dir: Path, filters: Filters, expected: set[str]
) -> None:
    with Searcher.open(index_dir) as searcher:
        dense = searcher.search("token", mode="dense", k=50, filters=filters, embedder=HashEmbedder)
        bm25 = searcher.search("token", mode="bm25", k=50, filters=filters)

    assert set(symbols(dense)) == expected
    assert set(symbols(bm25)) == expected


def test_large_path_filters_fall_back_to_post_filtering(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(filters_module, "MAX_IN_FILTER", 0)
    with Searcher.open(index_dir) as searcher:
        results = searcher.search(
            "token", mode="dense", k=3, filters=Filters(path_glob="src/*"), embedder=HashEmbedder
        )

    assert results
    assert all(r.chunk.file_path.startswith("src/") for r in results)


def test_glob_matching_nothing_returns_no_results(index_dir: Path) -> None:
    with Searcher.open(index_dir) as searcher:
        nothing = Filters(path_glob="docs/*")
        assert searcher.search("token", mode="bm25", filters=nothing) == []
        assert searcher.search("token", mode="dense", filters=nothing, embedder=HashEmbedder) == []


def test_stale_build_is_refused(make_repo: GitRepoFactory, tmp_path: Path) -> None:
    repo = make_repo(FILES)
    build(repo, tmp_path / "index", "window")
    commit_files(repo, {"src/config.py": "def load_config(path):\n    return None\n"})
    build(repo, tmp_path / "index", "treesitter")

    with pytest.raises(StaleIndexError, match="re-run `codebase-qa index --chunker window"):
        Searcher.open(tmp_path / "index", chunker="window")
    with Searcher.open(tmp_path / "index") as searcher:
        assert searcher.build.chunker == "treesitter"


def test_missing_index_is_reported(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError, match="run `codebase-qa index` first"):
        Searcher.open(tmp_path / "nowhere")
