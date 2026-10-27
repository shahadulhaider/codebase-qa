"""Read-only access to a repository's committed tree through the git CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class TreeEntry:
    path: str
    blob_sha: str
    size: int


class GitRepo:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            top = self._run("rev-parse", "--show-toplevel").strip()
        except GitError as error:
            raise GitError(f"{path} is not a git repository") from error
        self.root = Path(top)

    def rev_parse(self, rev: str = "HEAD") -> str:
        """Resolve ``rev`` to a full commit SHA."""
        try:
            return self._run("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").strip()
        except GitError:
            if rev == "HEAD" and not self._has_commits():
                raise GitError(f"{self.root} has no commits yet") from None
            raise GitError(f"unknown revision {rev!r} in {self.root}") from None

    def ls_tree(self, commit: str) -> list[TreeEntry]:
        """Regular files in ``commit``; symlinks and submodules are skipped."""
        output = self._run("ls-tree", "-r", "-z", "--long", commit)
        entries: list[TreeEntry] = []
        for record in output.split("\0"):
            if not record:
                continue
            meta, path = record.split("\t", 1)
            mode, kind, sha, size = meta.split()
            if kind == "blob" and mode in {"100644", "100755"}:
                entries.append(TreeEntry(path, sha, int(size)))
        return entries

    def read_blobs(self, shas: Iterable[str]) -> Iterator[tuple[str, bytes]]:
        """Stream blob contents through a single ``git cat-file --batch`` process."""
        wanted = list(dict.fromkeys(shas))
        if not wanted:
            return
        process = subprocess.Popen(
            ["git", "-C", str(self.root), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            for sha in wanted:
                process.stdin.write(f"{sha}\n".encode())
                process.stdin.flush()
                header = process.stdout.readline().decode().split()
                if len(header) != 3 or header[1] != "blob":
                    raise GitError(f"cannot read blob {sha}: {' '.join(header)}")
                content = process.stdout.read(int(header[2]))
                process.stdout.read(1)  # trailing newline
                yield sha, content
        finally:
            process.stdin.close()
            process.wait()

    def _has_commits(self) -> bool:
        try:
            return bool(self._run("rev-list", "-n", "1", "--all").strip())
        except GitError:
            return False

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.path), *args], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise GitError(result.stderr.strip() or f"git {args[0]} failed")
        return result.stdout
