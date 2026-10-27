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
    """Text for the FTS index: the original text plus identifier sub-words.

    The FTS tokenizer keeps ``_`` inside tokens, so ``get_user_by_id`` and
    ``getUserById`` each stay one exact token; their parts are appended so
    partial-word queries still match.
    """
    extra: list[str] = []
    for identifier in sorted(set(_IDENTIFIER.findall(text))):
        parts = split_identifier(identifier)
        if len(parts) > 1:
            extra.extend(parts)
    return text if not extra else f"{text}\n{' '.join(extra)}"


def symbol_terms(symbol_path: str) -> str:
    """FTS text for a qualified symbol: ``Client.getUser`` -> ``Client getUser client get user``."""
    if not symbol_path:
        return ""
    names = symbol_path.split(".")
    parts = [part for name in names for part in split_identifier(name)]
    return " ".join([*names, *parts])
