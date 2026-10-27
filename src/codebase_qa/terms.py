"""Code-aware term expansion for full-text search."""

from __future__ import annotations

import re

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+|[A-Z]+")


def split_identifier(identifier: str) -> list[str]:
    """``getHTTPResponse_code`` -> ``["get", "http", "response", "code"]``."""
    parts: list[str] = []
    for piece in identifier.split("_"):
        parts.extend(match.lower() for match in _CAMEL_PARTS.findall(piece))
    return parts


def search_terms(text: str) -> str:
    """Text for the FTS index: the original text plus camelCase sub-words.

    SQLite's ``unicode61`` tokenizer already splits on underscores and
    punctuation but keeps ``getUserById`` as one token, so its parts are
    appended. The original token stays searchable for exact-name queries.
    """
    extra: list[str] = []
    for identifier in set(_IDENTIFIER.findall(text)):
        parts = split_identifier(identifier)
        if len(parts) > 1 and any(c.isupper() for c in identifier):
            extra.extend(parts)
    return text if not extra else f"{text}\n{' '.join(extra)}"
