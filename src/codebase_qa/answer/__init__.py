"""Answer questions from retrieved chunks with validated ``[n]`` citations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from codebase_qa.answer.backend import AnswerBackend, BackendError, BackendUnavailableError
from codebase_qa.answer.citations import CitationFilter, check_citations
from codebase_qa.answer.openai_compat import (
    DEFAULT_BASE_URL,
    DEFAULT_LLM_MODEL,
    OpenAICompatibleBackend,
)
from codebase_qa.answer.prompt import SYSTEM_PROMPT, format_context, user_message
from codebase_qa.retrieval import SearchResult


class AnswerStream:
    """Iterate to stream the answer; citations to unknown sources are removed on the way.

    After iteration, :attr:`text` holds the full cleaned answer and
    :attr:`cited` / :attr:`invalid` the source numbers it referred to.
    """

    def __init__(
        self, question: str, sources: Sequence[SearchResult], backend: AnswerBackend
    ) -> None:
        self.question = question
        self.sources = list(sources)
        self.backend = backend
        self.system = SYSTEM_PROMPT
        self.user = user_message(question, self.sources)
        self._citations = CitationFilter(len(self.sources))
        self._parts: list[str] = []

    def __iter__(self) -> Iterator[str]:
        for piece in self.backend.stream(self.system, self.user):
            if text := self._citations.feed(piece):
                self._parts.append(text)
                yield text
        if text := self._citations.flush():
            self._parts.append(text)
            yield text

    @property
    def text(self) -> str:
        return "".join(self._parts)

    @property
    def cited(self) -> list[int]:
        return self._citations.cited

    @property
    def invalid(self) -> list[int]:
        return self._citations.invalid


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_LLM_MODEL",
    "SYSTEM_PROMPT",
    "AnswerBackend",
    "AnswerStream",
    "BackendError",
    "BackendUnavailableError",
    "CitationFilter",
    "OpenAICompatibleBackend",
    "check_citations",
    "format_context",
    "user_message",
]
