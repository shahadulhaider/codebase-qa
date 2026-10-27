from pathlib import Path

import pytest
from conftest import GitRepoFactory, commit_files

from codebase_qa.chunking import ChunkerName, whitespace_token_count
from codebase_qa.embeddings import HashEmbedder
from codebase_qa.indexer import (
    VECTORS_DIR,
    IndexOptions,
    IndexReport,
    collection_name,
    index_repo,
    index_stats,
)
from codebase_qa.vector_index import ChromaVectorIndex

FILES = {
    "src/client.py": (
        "class Client:\n"
        "    def get(self, url):\n"
        "        return url\n\n"
        "def load_config(path):\n"
        "    return path\n"
    ),
    "web/user.ts": "export function getUserById(id: number) {\n  return id;\n}\n",
    "README.md": "# Demo\n\nA small repository.\n",
    "logo.png": "not really an image",
    "uv.lock": "lockfile",
}


@pytest.fixture
def repo(make_repo: GitRepoFactory) -> Path:
    return make_repo(FILES)


def run(
    repo: Path, index_dir: Path, chunker: ChunkerName = "treesitter", rev: str = "HEAD"
) -> IndexReport:
    options = IndexOptions(chunker=chunker)
    return index_repo(repo, index_dir, HashEmbedder(), whitespace_token_count, options, rev=rev)


def test_full_index_writes_catalog_and_vectors(repo: Path, tmp_path: Path) -> None:
    index_dir = tmp_path / "index"

    report = run(repo, index_dir)

    assert (report.files, report.skipped) == (3, 2)
    assert report.embedded == report.chunks > 0
    collection = ChromaVectorIndex.open(index_dir / VECTORS_DIR, report.collection)
    assert collection.count() == report.chunks
    stats = index_stats(index_dir)
    assert stats.current_rev == report.commit
    assert stats.chunks["treesitter"] == report.chunks
    assert stats.languages == {"python": 3, "typescript": 1, "markdown": 1}
    assert [(b.vectors, b.stale) for b in stats.builds] == [(report.chunks, False)]


def test_vectors_carry_full_metadata(repo: Path, tmp_path: Path) -> None:
    report = run(repo, tmp_path / "index")
    collection = ChromaVectorIndex.open(tmp_path / "index" / VECTORS_DIR, report.collection)

    hits = collection.query(HashEmbedder().embed_query("load config path"), k=1)

    assert hits[0].metadata == {
        "file_path": "src/client.py",
        "language": "python",
        "symbol_name": "load_config",
        "symbol_path": "load_config",
        "symbol_kind": "function",
        "start_line": 5,
        "end_line": 6,
        "chunker": "treesitter",
        "commit": report.commit,
        "content_sha": hits[0].metadata["content_sha"],
    }


def test_rerun_at_same_revision_is_a_no_op(repo: Path, tmp_path: Path) -> None:
    run(repo, tmp_path / "index")

    second = run(repo, tmp_path / "index")

    assert second.up_to_date
    assert second.embedded == 0


def test_two_chunkers_coexist_in_one_index(repo: Path, tmp_path: Path) -> None:
    trees = run(repo, tmp_path / "index", "treesitter")
    windows = run(repo, tmp_path / "index", "window")

    stats = index_stats(tmp_path / "index")
    assert stats.chunks == {"treesitter": trees.chunks, "window": windows.chunks}
    assert {b.build.chunker for b in stats.builds} == {"treesitter", "window"}


def test_new_revision_marks_other_builds_stale(repo: Path, tmp_path: Path) -> None:
    run(repo, tmp_path / "index", "window")
    commit_files(repo, {"src/client.py": "def other():\n    pass\n"})

    report = run(repo, tmp_path / "index", "treesitter")

    stale = {b.build.chunker: b.stale for b in index_stats(tmp_path / "index").builds}
    assert stale == {"treesitter": False, "window": True}
    assert report.commit == index_stats(tmp_path / "index").current_rev


def test_indexes_an_older_revision(repo: Path, tmp_path: Path) -> None:
    commit_files(repo, {"src/extra.py": "def extra():\n    pass\n"})

    report = run(repo, tmp_path / "index", rev="HEAD~1")

    assert report.files == 3


def test_collection_name_is_valid_and_includes_revision() -> None:
    name = collection_name("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4", "window")

    assert name == "all-minilm-l6-v2-1110a243-window"
    assert collection_name("hash-64", None, "treesitter") == "hash-64-unpinned-treesitter"
