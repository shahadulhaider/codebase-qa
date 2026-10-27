"""Syntax-aware chunking with tree-sitter.

Every line of a file is owned by at most one emitted definition: the
innermost class, function or method that is not itself nested inside a
function. Classes own their header and fields, methods own their bodies,
nested functions stay inside their parent, and unowned lines become
``module`` chunks. Each owner's lines are then packed into token-budgeted
windows, so an oversized symbol is split but keeps its metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from tree_sitter import Node, Parser, QueryCursor

from codebase_qa.chunking.base import (
    Chunk,
    ChunkerName,
    Line,
    SymbolKind,
    TokenCounter,
    split_to_budget,
)
from codebase_qa.chunking.languages import LanguageRegistry, LanguageSpec

_CONTAINER_KINDS = frozenset({"class", "interface", "enum"})


@dataclass(eq=False)
class _Definition:
    name: str
    kind: SymbolKind
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parent: _Definition | None = None
    children: list[_Definition] = field(default_factory=lambda: list[_Definition]())

    @property
    def symbol_path(self) -> str:
        names: list[str] = []
        node: _Definition | None = self
        while node is not None:
            names.append(node.name)
            node = node.parent
        return ".".join(reversed(names))


class TreeSitterChunker:
    name: ChunkerName = "treesitter"

    def __init__(
        self,
        token_count: TokenCounter,
        *,
        registry: LanguageRegistry | None = None,
        max_tokens: int = 256,
        overlap_lines: int = 2,
    ) -> None:
        self._registry = registry or LanguageRegistry.default()
        self._token_count = token_count
        self._max_tokens = max_tokens
        self._overlap_lines = overlap_lines

    def supports(self, file_path: str) -> bool:
        return self._registry.spec_for(file_path) is not None

    def chunk(self, file_path: str, language: str, source: str) -> list[Chunk]:
        spec = self._registry.spec_for(file_path)
        if spec is None:
            raise ValueError(f"no tree-sitter grammar registered for {file_path}")
        lines = [text.rstrip("\r") for text in source.split("\n")]
        tree = Parser(spec.grammar).parse(source.encode())
        emitted = _emitted(_nest(_definitions(spec, tree.root_node)))

        owner: list[_Definition | None] = [None] * (len(lines) + 1)
        for definition in emitted:  # parents precede children, so children win
            for number in range(definition.start_line, definition.end_line + 1):
                if number <= len(lines):
                    owner[number] = definition

        chunks: list[Chunk] = []
        for run_owner, run in _runs(owner, lines):
            chunks.extend(self._pack(file_path, spec.language, run_owner, run))
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    def _pack(
        self,
        file_path: str,
        language: str,
        definition: _Definition | None,
        run: list[Line],
    ) -> list[Chunk]:
        if definition is None:
            kind: SymbolKind = "module"
            name = path = ""
        else:
            kind, name, path = definition.kind, definition.name, definition.symbol_path
        header = Chunk(file_path, language, "", 1, 1, kind, self.name, name, path).header
        windows = split_to_budget(
            run,
            header=header,
            token_count=self._token_count,
            max_tokens=self._max_tokens,
            overlap_lines=self._overlap_lines,
        )
        return [
            Chunk(
                file_path=file_path,
                language=language,
                text="\n".join(line.text for line in _trim(window)),
                start_line=_trim(window)[0].number,
                end_line=_trim(window)[-1].number,
                symbol_kind=kind,
                chunker=self.name,
                symbol_name=name,
                symbol_path=path,
            )
            for window in windows
            if _trim(window)
        ]


def _definitions(spec: LanguageSpec, root: Node) -> list[_Definition]:
    found: dict[tuple[int, int], _Definition] = {}
    for _, captures in QueryCursor(spec.query).matches(root):
        name_nodes = captures.get("name", [])
        for capture, nodes in captures.items():
            if not capture.startswith("definition.") or not nodes or not name_nodes:
                continue
            node = _expand(nodes[0], spec)
            key = (node.start_byte, node.end_byte)
            if key in found:
                continue
            start = _leading_comments_start(node, spec)
            found[key] = _Definition(
                name=(name_nodes[0].text or b"").decode(errors="replace"),
                kind=cast(SymbolKind, capture.removeprefix("definition.")),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=start.start_point.row + 1,
                end_line=_end_line(node),
            )
    return list(found.values())


def _expand(node: Node, spec: LanguageSpec) -> Node:
    """Grow a definition node to include wrappers such as decorators and ``export``."""
    while node.parent is not None and node.parent.type in spec.wrapper_types:
        node = node.parent
    return node


def _leading_comments_start(node: Node, spec: LanguageSpec) -> Node:
    """Attach comments that directly precede a definition (e.g. JSDoc)."""
    start = node
    previous = node.prev_named_sibling
    while (
        previous is not None
        and previous.type in spec.comment_types
        and previous.end_point.row >= start.start_point.row - 1
    ):
        start = previous
        previous = previous.prev_named_sibling
    return start


def _end_line(node: Node) -> int:
    row, column = node.end_point
    return max(node.start_point.row + 1, row + 1 if column > 0 else row)


def _nest(definitions: list[_Definition]) -> list[_Definition]:
    definitions.sort(key=lambda d: (d.start_byte, -d.end_byte))
    roots: list[_Definition] = []
    stack: list[_Definition] = []
    for definition in definitions:
        while stack and stack[-1].end_byte <= definition.start_byte:
            stack.pop()
        if stack:
            definition.parent = stack[-1]
            stack[-1].children.append(definition)
        else:
            roots.append(definition)
        stack.append(definition)
    return roots


def _emitted(roots: list[_Definition]) -> list[_Definition]:
    """Definitions that get their own chunks, parents before children.

    Members of classes are emitted; anything nested inside a function is
    absorbed into that function. Adjacent same-named siblings (overload
    signatures, property getter/setter pairs) are merged into one definition.
    """
    result: list[_Definition] = []

    def visit(siblings: list[_Definition], parent: _Definition | None) -> None:
        for definition in _merge_overloads(siblings):
            definition.parent = parent
            if definition.kind == "function" and parent is not None and parent.kind == "class":
                definition.kind = "method"
            if definition.kind == "method" and (parent is None or parent.kind != "class"):
                definition.kind = "function"
            result.append(definition)
            if definition.kind in _CONTAINER_KINDS:
                visit(definition.children, definition)

    visit(roots, None)
    return result


def _merge_overloads(siblings: list[_Definition]) -> list[_Definition]:
    merged: list[_Definition] = []
    for definition in siblings:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.name == definition.name
            and previous.kind == definition.kind
            and definition.start_line <= previous.end_line + 2
        ):
            previous.end_byte = definition.end_byte
            previous.end_line = definition.end_line
            previous.children.extend(definition.children)
            continue
        merged.append(definition)
    return merged


def _runs(
    owner: list[_Definition | None], lines: list[str]
) -> list[tuple[_Definition | None, list[Line]]]:
    runs: list[tuple[_Definition | None, list[Line]]] = []
    for number in range(1, len(lines) + 1):
        line = Line(number, lines[number - 1])
        current = owner[number]
        if runs and runs[-1][0] is current and runs[-1][1][-1].number == number - 1:
            runs[-1][1].append(line)
        else:
            runs.append((current, [line]))
    return [(o, run) for o, run in runs if any(_has_content(line.text) for line in run)]


def _has_content(text: str) -> bool:
    """False for blank lines and bare closing punctuation such as ``}``."""
    return bool(text.strip().strip("{}()[];,"))


def _trim(lines: list[Line]) -> list[Line]:
    start, end = 0, len(lines)
    while start < end and not lines[start].text.strip():
        start += 1
    while end > start and not lines[end - 1].text.strip():
        end -= 1
    return lines[start:end]
