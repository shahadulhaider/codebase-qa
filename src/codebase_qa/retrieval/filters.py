"""Resolve filters once so every retriever sees the same candidate set."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from codebase_qa.retrieval.types import Filters
from codebase_qa.store import Store

MAX_IN_FILTER = 2_000
"""Above this many files, the dense retriever post-filters an oversampled result."""


@dataclass(frozen=True)
class ResolvedFilters:
    filters: Filters
    files: tuple[str, ...] | None
    """Files matching ``path_glob``, or None when there is no path constraint."""

    @classmethod
    def resolve(cls, store: Store, filters: Filters) -> ResolvedFilters:
        files = None
        if filters.path_glob is not None:
            files = tuple(store.matching_files(filters.path_glob))
        return cls(filters, files)

    @property
    def empty(self) -> bool:
        """True when the path constraint matches nothing, so no search can succeed."""
        return self.files is not None and not self.files

    @property
    def pushes_files_to_vector_store(self) -> bool:
        return self.files is not None and len(self.files) <= MAX_IN_FILTER

    def chroma_where(self) -> dict[str, Any] | None:
        """Chroma ``where`` clause; Chroma requires ``$and`` for two or more terms."""
        terms: list[dict[str, Any]] = []
        if self.filters.languages:
            terms.append({"language": {"$in": list(self.filters.languages)}})
        if self.filters.kinds:
            terms.append({"symbol_kind": {"$in": list(self.filters.kinds)}})
        if self.filters.roles:
            terms.append({"role": {"$in": list(self.filters.roles)}})
        if self.files is not None and self.pushes_files_to_vector_store:
            terms.append({"file_path": {"$in": list(self.files)}})
        if not terms:
            return None
        return terms[0] if len(terms) == 1 else {"$and": terms}

    @cached_property
    def _file_set(self) -> frozenset[str]:
        return frozenset(self.files or ())

    def accepts_file(self, path: str) -> bool:
        return self.files is None or path in self._file_set
