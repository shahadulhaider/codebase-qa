"""BM25 retrieval over the catalog's FTS5 table."""

from __future__ import annotations

import re

from codebase_qa.chunking import ChunkerName
from codebase_qa.retrieval.filters import ResolvedFilters
from codebase_qa.store import Store
from codebase_qa.terms import split_identifier

_WORD = re.compile(r"[A-Za-z0-9_]+")


def fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 expression: quoted terms joined by OR.

    Raw user text is never passed to MATCH, where ``-``, ``:`` and quotes
    are operators. Identifiers contribute the whole token and its parts.
    """
    terms: list[str] = []
    for word in _WORD.findall(text):
        terms.append(word.lower())
        parts = split_identifier(word)
        if len(parts) > 1:
            terms.extend(parts)
    unique = [t for t in dict.fromkeys(terms) if t.strip("_")]
    if not unique:
        return None
    return " OR ".join(f'"{t}"' for t in unique)


def bm25_search(
    store: Store, chunker: ChunkerName, query: str, k: int, filters: ResolvedFilters
) -> list[tuple[str, float]]:
    match = fts_query(query)
    if match is None or filters.empty:
        return []
    return store.bm25(
        chunker,
        match,
        k,
        languages=filters.filters.languages,
        kinds=filters.filters.kinds,
        roles=filters.filters.roles,
        files=filters.files,
    )
