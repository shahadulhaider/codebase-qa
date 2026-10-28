"""Answer backend for the Anthropic Messages API (optional ``anthropic`` extra)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from codebase_qa.answer.backend import BackendError, BackendUnavailableError

if TYPE_CHECKING:
    from anthropic import Anthropic

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
"""Models that accept server-side refusal fallbacks (``fallbacks: "default"``)."""
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicBackend:
    """Streams an answer from Claude.

    The API key comes from the SDK's own resolution (``ANTHROPIC_API_KEY`` or
    an ``ant auth login`` profile) unless ``api_key`` is given.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16000,
    ) -> None:
        try:
            import anthropic
        except ImportError as error:
            raise BackendError(
                "the anthropic backend needs the optional dependency: uv sync --extra anthropic"
            ) from error
        self.model = model
        self._max_tokens = max_tokens
        self._client: Anthropic = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def stream(self, system: str, user: str) -> Iterator[str]:
        import anthropic

        fallbacks = self.model in _FALLBACK_MODELS
        try:
            with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                **({"betas": [_FALLBACK_BETA], "fallbacks": "default"} if fallbacks else {}),  # pyright: ignore[reportArgumentType]
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except TypeError as error:
            # The SDK resolves credentials lazily and signals their absence with a TypeError.
            if "authentication" not in str(error):
                raise
            raise BackendError(
                "no Anthropic credentials: set ANTHROPIC_API_KEY or CODEBASE_QA_API_KEY"
            ) from error
        except anthropic.APIConnectionError as error:
            raise BackendUnavailableError(f"cannot reach the Anthropic API: {error}") from error
        except anthropic.APIStatusError as error:
            raise BackendError(
                f"Anthropic API error {error.status_code}: {error.message}"
            ) from error
        if final.stop_reason == "refusal":
            raise BackendError("the model declined to answer (stop_reason: refusal)")
        if final.stop_reason == "max_tokens":
            raise BackendError(f"the answer was cut off at max_tokens={self._max_tokens}")
