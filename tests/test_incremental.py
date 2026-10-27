from pathlib import Path

import pytest
from conftest import GitRepoFactory, commit_files, git

from codebase_qa.chunking import ChunkerName, whitespace_token_count
from codebase_qa.embeddings import HashEmbedder
from codebase_qa.indexer import (
    VECTORS_DIR,
    IndexOptions,
    IndexReport,
    index_repo,
    index_stats,
    is_up_to_date,
)
from codebase_qa.vector_index import ChromaVectorIndex

MODULE = """def alpha():
    return "alpha"


def beta():
    return "beta"


def gamma():
    return "gamma"
"""


@pytest.fixture
def repo(make_repo: GitRepoFactory) -> Path:
    return make_repo({"pkg/mod.py": MODULE, "pkg/other.py": "def other():\n    return 1\n"})


class Indexer:
    def __init__(self, repo: Path, index_dir: Path) -> None:
        self.repo = repo
        self.index_dir = index_dir
        self.embedder = HashEmbedder()

    def run(self, chunker: ChunkerName = "treesitter", **options: object) -> IndexReport:
        before = self.embedder.documents_embedded
        report = index_repo(
            self.repo,
            self.index_dir,
            self.embedder,
            whitespace_token_count,
            IndexOptions(chunker=chunker, **options),  # type: ignore[arg-type]
        )
        assert self.embedder.documents_embedded - before == report.sync.embedded
        return report

    def collection(self, report: IndexReport) -> ChromaVectorIndex:
        return ChromaVectorIndex.open(self.index_dir / VECTORS_DIR, report.collection)

    def assert_in_sync(self, report: IndexReport) -> None:
        ids = set(self.collection(report).content_shas())
        assert len(ids) == report.chunks


@pytest.fixture
def indexer(repo: Path, tmp_path: Path) -> Indexer:
    return Indexer(repo, tmp_path / "index")


def test_editing_one_function_embeds_only_that_chunk(repo: Path, indexer: Indexer) -> None:
    first = indexer.run()
    commit_files(repo, {"pkg/mod.py": MODULE.replace('"beta"', '"BETA"')})

    second = indexer.run()

    assert (second.changes.added, second.changes.modified, second.changes.unchanged) == (0, 1, 1)
    assert second.sync.embedded == 1
    assert second.sync.removed == 1
    assert second.chunks == first.chunks
    indexer.assert_in_sync(second)


def test_unchanged_revision_embeds_nothing(indexer: Indexer) -> None:
    indexer.run()

    again = indexer.run()

    assert again.up_to_date
    assert again.sync.embedded == 0


def test_shifted_lines_reuse_vectors_instead_of_reembedding(repo: Path, indexer: Indexer) -> None:
    indexer.run()
    commit_files(repo, {"pkg/mod.py": "import os\n\n\n" + MODULE})

    report = indexer.run()

    # Every function moved down three lines: new chunk ids, same embedded text.
    assert report.sync.embedded == 1  # the new module chunk holding the import
    assert report.sync.reused == 3
    hit = indexer.collection(report).query(HashEmbedder().embed_query("gamma"), k=1)[0]
    assert (hit.metadata["symbol_name"], hit.metadata["start_line"]) == ("gamma", 12)


def test_deleted_file_removes_its_vectors(repo: Path, indexer: Indexer) -> None:
    indexer.run()
    (repo / "pkg/other.py").unlink()
    git(repo, "commit", "-q", "-am", "remove other")

    report = indexer.run()

    assert report.changes.deleted == 1
    assert report.sync.removed == 1
    names = {
        m["symbol_name"]
        for m in (
            h.metadata
            for h in indexer.collection(report).query(HashEmbedder().embed_query("x"), k=10)
        )
    }
    assert "other" not in names
    indexer.assert_in_sync(report)


def test_kept_chunks_get_the_new_commit_metadata(repo: Path, indexer: Indexer) -> None:
    indexer.run()
    commit_files(repo, {"pkg/new.py": "def delta():\n    return 4\n"})

    report = indexer.run()

    hit = indexer.collection(report).query(HashEmbedder().embed_query("alpha"), k=1)[0]
    assert hit.metadata["commit"] == report.commit
    assert report.sync.kept == 4


def test_stale_build_catches_up_without_reembedding_unchanged_text(
    repo: Path, indexer: Indexer
) -> None:
    indexer.run("window")
    indexer.run("treesitter")
    commit_files(repo, {"pkg/mod.py": MODULE.replace('"gamma"', '"GAMMA"')})
    indexer.run("treesitter")
    assert [
        b.stale for b in index_stats(indexer.index_dir).builds if b.build.chunker == "window"
    ] == [True]

    caught_up = indexer.run("window")

    assert caught_up.changes.modified == 0  # catalog was already at this revision
    assert caught_up.sync.embedded == 1  # only the window containing the edit
    assert all(not b.stale for b in index_stats(indexer.index_dir).builds)
    indexer.assert_in_sync(caught_up)


def test_changed_chunk_settings_force_rechunking(indexer: Indexer) -> None:
    first = indexer.run()

    rechunked = indexer.run(max_tokens=3)

    assert not rechunked.up_to_date
    assert rechunked.chunks > first.chunks
    indexer.assert_in_sync(rechunked)


def test_verify_repairs_vectors_missing_from_the_collection(indexer: Indexer) -> None:
    report = indexer.run()
    collection = indexer.collection(report)
    lost = sorted(collection.content_shas())[:2]
    collection.delete(lost)

    repaired = indexer.run(verify=True)

    assert repaired.sync.embedded == 2
    indexer.assert_in_sync(repaired)


def test_verify_removes_orphan_vectors(indexer: Indexer) -> None:
    report = indexer.run()
    collection = indexer.collection(report)
    orphan = HashEmbedder().embed_documents(["orphan"])
    collection.upsert(["orphan"], orphan, ["orphan"], [{"content_sha": "x"}])

    repaired = indexer.run(verify=True)

    assert repaired.sync.removed == 1
    indexer.assert_in_sync(repaired)


def test_is_up_to_date_is_checked_without_a_model(repo: Path, indexer: Indexer) -> None:
    options = IndexOptions()
    embedder = HashEmbedder()
    assert not is_up_to_date(repo, indexer.index_dir, embedder.model_id, None, options)

    indexer.run()

    assert is_up_to_date(repo, indexer.index_dir, embedder.model_id, None, options)
    assert not is_up_to_date(
        repo, indexer.index_dir, embedder.model_id, None, IndexOptions(verify=True)
    )
