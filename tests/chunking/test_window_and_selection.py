from pathlib import Path

import pytest

from codebase_qa.chunking import (
    Chunk,
    FileRoute,
    LanguageRegistry,
    LineWindowChunker,
    chunk_file,
    route_file,
    whitespace_token_count,
)
from codebase_qa.chunking.selection import role_for_path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "chunking"
REGISTRY = LanguageRegistry.default()


@pytest.mark.parametrize(
    ("name", "language"),
    [
        ("README.md", "markdown"),
        ("doc.rst", "rst"),
        ("config.json", "json"),
        ("config.yaml", "yaml"),
        ("config.toml", "toml"),
        ("page.html", "html"),
        ("style.css", "css"),
    ],
)
def test_text_files_route_to_windows(name: str, language: str) -> None:
    route = route_file(name, 100, registry=REGISTRY)
    assert route == FileRoute(language, syntax_aware=False)
    assert route is not None

    chunks = chunk_file(
        name,
        (FIXTURES / name).read_text(),
        route,
        chunker="treesitter",
        token_count=whitespace_token_count,
    )

    assert [(c.symbol_kind, c.language, c.start_line, c.chunker) for c in chunks] == [
        ("window", language, 1, "treesitter")
    ]


@pytest.mark.parametrize(
    ("path", "language"),
    [("a.py", "python"), ("a.ts", "typescript"), ("a.tsx", "typescript"), ("a.jsx", "javascript")],
)
def test_code_files_route_to_tree_sitter(path: str, language: str) -> None:
    assert route_file(path, 100, registry=REGISTRY) == FileRoute(language, syntax_aware=True)


@pytest.mark.parametrize(
    ("path", "size", "excludes"),
    [
        ("main.go", 10, ()),
        ("big.py", 2_000_000, ()),
        ("web/node_modules/x/index.js", 10, ()),
        ("dist/app.js", 10, ()),
        ("package-lock.json", 10, ()),
        ("uv.lock", 10, ()),
        ("static/app.min.js", 10, ()),
        ("tests/fixtures/data.json", 10, ("tests/*",)),
    ],
)
def test_skipped_files(path: str, size: int, excludes: tuple[str, ...]) -> None:
    assert route_file(path, size, registry=REGISTRY, excludes=excludes) is None


def test_window_chunker_baseline_ignores_syntax() -> None:
    source = (FIXTURES / "sample.py").read_text()
    route = route_file("sample.py", len(source), registry=REGISTRY)
    assert route is not None

    chunks = chunk_file(
        "sample.py",
        source,
        route,
        chunker="window",
        token_count=whitespace_token_count,
        max_tokens=40,
    )

    assert len(chunks) > 1
    assert {(c.chunker, c.symbol_kind, c.symbol_path) for c in chunks} == {("window", "window", "")}
    assert chunks[0].start_line == 1


def test_window_chunker_skips_blank_files() -> None:
    assert LineWindowChunker(whitespace_token_count).chunk("x.md", "markdown", "\n\n  \n") == []


def test_identity_depends_on_location_but_content_sha_on_embedded_text() -> None:
    a = Chunk("a.py", "python", "def f(): pass", 1, 1, "function", "treesitter", "f", "f")
    moved = Chunk("a.py", "python", "def f(): pass", 10, 10, "function", "treesitter", "f", "f")
    renamed = Chunk("b.py", "python", "def f(): pass", 1, 1, "function", "treesitter", "f", "f")

    assert a.chunk_id != moved.chunk_id
    assert a.content_sha == moved.content_sha
    assert a.content_sha != renamed.content_sha
    assert a.embedding_text == "# a.py\n# f (function)\ndef f(): pass"


@pytest.mark.parametrize(
    ("path", "role"),
    [
        ("httpx/_client.py", "source"),
        ("tests/client/test_redirects.py", "test"),
        ("pkg/test_utils.py", "test"),
        ("pkg/utils_test.py", "test"),
        ("conftest.py", "test"),
        ("web/src/button.test.tsx", "test"),
        ("web/src/__tests__/button.tsx", "test"),
        ("tests/fixtures/data.json", "test"),
        ("docs/advanced/transports.md", "docs"),
        ("README.md", "docs"),
        ("docs/conf.py", "docs"),
        ("pyproject.toml", "config"),
        ("web/tsconfig.json", "config"),
        ("contest.py", "source"),
    ],
)
def test_role_for_path(path: str, role: str) -> None:
    assert role_for_path(path) == role


def test_overlong_single_line_splits_at_word_boundaries() -> None:
    words = [f"word{i}" for i in range(300)]
    source = "# Title\n\n" + " ".join(words) + "\n\nAfter.\n"
    chunker = LineWindowChunker(whitespace_token_count, max_tokens=50, overlap_lines=0)

    chunks = chunker.chunk("docs/long.md", "markdown", source)

    pieces = [c for c in chunks if c.start_line == 3]
    assert len(pieces) > 1
    assert all(whitespace_token_count(c.embedding_text) <= 50 for c in chunks)
    rebuilt = " ".join(c.text.split("\n")[0] for c in pieces).split()
    assert rebuilt == words
