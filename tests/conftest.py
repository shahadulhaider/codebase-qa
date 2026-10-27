import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

GitRepoFactory = Callable[[dict[str, str]], Path]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def commit_files(repo: Path, files: dict[str, str], message: str = "update") -> str:
    for path, content in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def make_repo(tmp_path: Path) -> GitRepoFactory:
    def factory(files: dict[str, str]) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        git(repo, "config", "user.email", "test@example.com")
        git(repo, "config", "user.name", "Test")
        if files:
            commit_files(repo, files, "initial")
        return repo

    return factory
