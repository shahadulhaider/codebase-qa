import hashlib
from pathlib import Path

import pytest
from conftest import GitRepoFactory, commit_files

from codebase_qa.git import GitError, GitRepo


def test_ls_tree_lists_blobs_with_content_hashes(make_repo: GitRepoFactory) -> None:
    repo = make_repo({"a.py": "x = 1\n", "pkg/b.ts": "export const y = 2;\n"})
    git = GitRepo(repo)

    entries = {e.path: e for e in git.ls_tree(git.rev_parse())}

    assert set(entries) == {"a.py", "pkg/b.ts"}
    content = b"x = 1\n"
    expected = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()
    assert entries["a.py"].blob_sha == expected
    assert entries["a.py"].size == len(content)


def test_read_blobs_streams_contents(make_repo: GitRepoFactory) -> None:
    repo = make_repo({"a.py": "x = 1\n", "b.py": "y = 2\n", "same.py": "x = 1\n"})
    git = GitRepo(repo)
    entries = git.ls_tree(git.rev_parse())

    contents = dict(git.read_blobs(e.blob_sha for e in entries))

    assert sorted(contents.values()) == [b"x = 1\n", b"y = 2\n"]


def test_rev_parse_resolves_branches_and_old_commits(make_repo: GitRepoFactory) -> None:
    repo = make_repo({"a.py": "1\n"})
    first = GitRepo(repo).rev_parse()
    commit_files(repo, {"a.py": "2\n"})
    git = GitRepo(repo)

    assert git.rev_parse("main") != first
    assert git.rev_parse("HEAD~1") == first


def test_empty_repository_has_clear_error(make_repo: GitRepoFactory) -> None:
    with pytest.raises(GitError, match="no commits yet"):
        GitRepo(make_repo({})).rev_parse()


def test_unknown_revision_has_clear_error(make_repo: GitRepoFactory) -> None:
    with pytest.raises(GitError, match="unknown revision 'nope'"):
        GitRepo(make_repo({"a.py": "1\n"})).rev_parse("nope")


def test_non_repository_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GitError, match="not a git repository"):
        GitRepo(tmp_path)
