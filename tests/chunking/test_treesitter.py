from itertools import pairwise
from pathlib import Path

import pytest

from codebase_qa.chunking import (
    Chunk,
    LanguageRegistry,
    LanguageSpec,
    TreeSitterChunker,
    whitespace_token_count,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "chunking"


def chunk(name: str) -> list[Chunk]:
    chunker = TreeSitterChunker(whitespace_token_count)
    return chunker.chunk(name, "unused", (FIXTURES / name).read_text())


def outline(chunks: list[Chunk]) -> list[tuple[str, str, int, int]]:
    return [(c.symbol_kind, c.symbol_path, c.start_line, c.end_line) for c in chunks]


def test_python_definitions() -> None:
    assert outline(chunk("sample.py")) == [
        ("module", "", 1, 6),
        ("function", "retry", 9, 16),
        ("class", "Client", 19, 24),
        ("method", "Client.get", 26, 28),
        ("method", "Client.close", 30, 31),
        ("method", "Client.name", 33, 39),
        ("module", "", 42, 43),
    ]


def test_python_decorators_and_nested_functions_stay_attached() -> None:
    by_path = {c.symbol_path: c for c in chunk("sample.py")}

    assert by_path["Client"].text.startswith("@dataclass\nclass Client:")
    assert "timeout: float = DEFAULT_TIMEOUT" in by_path["Client"].text
    assert by_path["Client.get"].text.lstrip().startswith("@retry(attempts=2)")
    assert "def wrapper" in by_path["retry"].text
    assert "wrapper" not in by_path


def test_typescript_definitions() -> None:
    assert outline(chunk("sample.ts")) == [
        ("module", "", 1, 1),
        ("interface", "User", 3, 6),
        ("type", "UserId", 8, 8),
        ("enum", "Role", 10, 13),
        ("function", "format", 15, 20),
        ("class", "UserService", 22, 23),
        ("method", "UserService.constructor", 25, 25),
        ("method", "UserService.getUser", 27, 30),
        ("function", "debounce", 33, 39),
    ]


def test_typescript_overloads_merge_and_jsdoc_attaches() -> None:
    fmt = next(c for c in chunk("sample.ts") if c.symbol_path == "format")

    assert fmt.text.startswith("/** Formats a value. */")
    assert fmt.text.count("export function format") == 3


def test_tsx_and_jsx() -> None:
    assert outline(chunk("Button.tsx")) == [("type", "Props", 1, 1), ("function", "Button", 3, 5)]
    assert outline(chunk("widget.jsx")) == [
        ("class", "Widget", 1, 1),
        ("method", "Widget.render", 2, 4),
        ("function", "helper", 7, 9),
    ]


def test_syntax_errors_still_yield_surrounding_definitions() -> None:
    names = [c.symbol_path for c in chunk("broken.py")]

    assert names[0] == "ok"
    assert names[-1] == "also_ok"


def test_oversized_symbol_splits_into_windows_with_metadata() -> None:
    body = "\n".join(f"    x{i} = {i}  # filler words here" for i in range(60))
    source = f"def big():\n{body}\n    return x0\n"
    chunker = TreeSitterChunker(whitespace_token_count, max_tokens=50, overlap_lines=1)

    parts = chunker.chunk("big.py", "python", source)

    assert len(parts) > 1
    assert {(c.symbol_kind, c.symbol_path) for c in parts} == {("function", "big")}
    assert parts[0].start_line == 1
    assert parts[-1].end_line == 62
    assert all(whitespace_token_count(c.embedding_text) <= 50 for c in parts)
    for previous, current in pairwise(parts):
        assert current.start_line == previous.end_line  # one line of overlap


def test_chunk_text_matches_cited_lines() -> None:
    source = (FIXTURES / "sample.ts").read_text().split("\n")
    for c in chunk("sample.ts"):
        assert c.text == "\n".join(source[c.start_line - 1 : c.end_line])


def test_new_language_is_registered_without_chunker_changes(tmp_path: Path) -> None:
    import tree_sitter_python

    spec = LanguageSpec(
        language="starlark",
        extensions=(".star",),
        load_grammar=tree_sitter_python.language,
        query_source="(function_definition name: (identifier) @name) @definition.function",
    )
    registry = LanguageRegistry([spec])
    chunker = TreeSitterChunker(whitespace_token_count, registry=registry)

    chunks = chunker.chunk("BUILD.star", "starlark", "def rule(ctx):\n    pass\n")

    assert outline(chunks) == [("function", "rule", 1, 2)]
    assert chunks[0].language == "starlark"


def test_unregistered_extension_is_rejected() -> None:
    with pytest.raises(ValueError, match="no tree-sitter grammar"):
        TreeSitterChunker(whitespace_token_count).chunk("main.go", "go", "package main")


def test_repeated_parsing_is_stable() -> None:
    # tree-sitter 0.26.0 corrupts memory under this access pattern and
    # segfaults; the dependency is pinned below it. A crash fails the run.
    chunker = TreeSitterChunker(whitespace_token_count)
    sources = sorted((FIXTURES.parent.parent.parent / "src").rglob("*.py"))
    sources += sorted(p for p in FIXTURES.iterdir() if chunker.supports(p.name))
    texts = [(p.as_posix(), p.read_text()) for p in sources]
    for _ in range(15):
        for path, text in texts:
            chunker.chunk(path, "unused", text)
