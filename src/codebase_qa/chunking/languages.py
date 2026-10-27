"""Language registry: file extensions, grammars and definition queries.

Adding a language means registering a ``LanguageSpec`` with its grammar
loader and a tree-sitter query file; the chunker itself is language-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path, PurePosixPath

from tree_sitter import Language, Query

GrammarLoader = Callable[[], object]
"""Returns the grammar's raw language pointer, e.g. ``tree_sitter_python.language()``."""


@dataclass(frozen=True)
class LanguageSpec:
    """How to find definitions in one language.

    The query must capture each definition node as ``@definition.<kind>``
    (kind is one of class, function, method, interface, type, enum) and its
    identifier as ``@name``. ``wrapper_types`` are parent node types that belong
    to the definition, such as decorators or ``export``.
    """

    language: str
    extensions: tuple[str, ...]
    load_grammar: GrammarLoader
    query_source: str
    wrapper_types: frozenset[str] = field(default_factory=frozenset[str])
    comment_types: frozenset[str] = frozenset({"comment"})

    @property
    def grammar(self) -> Language:
        return _language(self.load_grammar)

    @property
    def query(self) -> Query:
        return _query(self.load_grammar, self.query_source)


@cache
def _language(load: GrammarLoader) -> Language:
    return Language(load())


@cache
def _query(load: GrammarLoader, source: str) -> Query:
    return Query(_language(load), source)


def _builtin_query(name: str) -> str:
    return resources.files("codebase_qa.chunking.queries").joinpath(name).read_text()


def _python() -> object:
    import tree_sitter_python

    return tree_sitter_python.language()


def _typescript() -> object:
    import tree_sitter_typescript

    return tree_sitter_typescript.language_typescript()


def _tsx() -> object:
    import tree_sitter_typescript

    return tree_sitter_typescript.language_tsx()


def _javascript() -> object:
    import tree_sitter_javascript

    return tree_sitter_javascript.language()


_JS_WRAPPERS = frozenset({"export_statement", "lexical_declaration", "variable_declaration"})


def builtin_specs() -> list[LanguageSpec]:
    return [
        LanguageSpec(
            "python",
            (".py", ".pyi"),
            _python,
            _builtin_query("python.scm"),
            frozenset({"decorated_definition"}),
        ),
        LanguageSpec(
            "typescript",
            (".ts", ".mts", ".cts"),
            _typescript,
            _builtin_query("typescript.scm"),
            _JS_WRAPPERS,
        ),
        LanguageSpec("typescript", (".tsx",), _tsx, _builtin_query("typescript.scm"), _JS_WRAPPERS),
        LanguageSpec(
            "javascript",
            (".js", ".mjs", ".cjs", ".jsx"),
            _javascript,
            _builtin_query("javascript.scm"),
            _JS_WRAPPERS,
        ),
    ]


WINDOW_LANGUAGES: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
}


class LanguageRegistry:
    def __init__(self, specs: Iterable[LanguageSpec] = ()) -> None:
        self._by_extension: dict[str, LanguageSpec] = {}
        for spec in specs:
            self.register(spec)

    @classmethod
    def default(cls) -> LanguageRegistry:
        return cls(builtin_specs())

    def register(self, spec: LanguageSpec) -> None:
        for extension in spec.extensions:
            self._by_extension[extension.lower()] = spec

    def spec_for(self, path: str | Path) -> LanguageSpec | None:
        return self._by_extension.get(PurePosixPath(path).suffix.lower())
