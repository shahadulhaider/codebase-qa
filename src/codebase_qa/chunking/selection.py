"""Which repository files are indexed, and how."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Literal

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


Role = Literal["source", "test", "docs", "config"]

_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs"})
_DOC_DIRS = frozenset({"doc", "docs"})
_TEST_FILES = ("test_*.py", "*_test.py", "conftest.py", "*.test.*", "*.spec.*")
_DOC_SUFFIXES = frozenset({".md", ".markdown", ".rst"})
_CONFIG_SUFFIXES = frozenset({".json", ".yaml", ".yml", ".toml"})


def role_for_path(path: str) -> Role:
    """Classify a repo-relative path; tests win over docs, docs over config.

    The rules are path conventions only, so a fixture under ``tests/`` is a
    test file whatever its extension.
    """
    posix = PurePosixPath(path)
    directories = {part.lower() for part in posix.parts[:-1]}
    if directories & _TEST_DIRS or any(fnmatch(posix.name, p) for p in _TEST_FILES):
        return "test"
    if directories & _DOC_DIRS or posix.suffix.lower() in _DOC_SUFFIXES:
        return "docs"
    if posix.suffix.lower() in _CONFIG_SUFFIXES:
        return "config"
    return "source"
