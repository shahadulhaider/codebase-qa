"""The interface every answer-generating model backend implements."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class BackendError(RuntimeError):
    """The backend rejected the request or returned something unusable."""


class BackendUnavailableError(BackendError):
    """The backend could not be reached at all (connection refused, DNS, timeout)."""


class AnswerBackend(Protocol):
    @property
    def name(self) -> str:
        """Short label for messages, e.g. ``openai-compatible``."""
        ...

    @property
    def model(self) -> str: ...

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the answer text in pieces as the model produces it.

        Raises :class:`BackendError` or :class:`BackendUnavailableError`; errors
        can surface on the first ``next()`` rather than at call time.
        """
        ...
