"""Which repository files are indexed, and how."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import PurePosixPath

from codebase_qa.chunking.languages import WINDOW_LANGUAGES, LanguageRegistry

MAX_FILE_BYTES = 1_000_000

EXCLUDED_DIRS = frozenset({".git", "node_modules", "dist", "build", "vendor", "__pycache__"})
EXCLUDED_FILES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
    }
)
EXCLUDED_PATTERNS = ("*.min.js", "*.min.css", "*.map")


@dataclass(frozen=True)
class FileRoute:
    language: str
    syntax_aware: bool


def route_file(
    path: str,
    size: int,
    *,
    registry: LanguageRegistry,
    excludes: Sequence[str] = (),
) -> FileRoute | None:
    """Return how to chunk ``path`` (repo-relative, POSIX), or None to skip it.

    User ``excludes`` are fnmatch patterns matched against the whole path, so
    ``*`` also crosses directory separators.
    """
    if size > MAX_FILE_BYTES:
        return None
    posix = PurePosixPath(path)
    if any(part in EXCLUDED_DIRS for part in posix.parts[:-1]):
        return None
    if posix.name in EXCLUDED_FILES or any(fnmatch(posix.name, p) for p in EXCLUDED_PATTERNS):
        return None
    if any(fnmatch(path, pattern) for pattern in excludes):
        return None
    spec = registry.spec_for(path)
    if spec is not None:
        return FileRoute(spec.language, syntax_aware=True)
    language = WINDOW_LANGUAGES.get(posix.suffix.lower())
    if language is not None:
        return FileRoute(language, syntax_aware=False)
    return None
