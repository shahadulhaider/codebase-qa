"""Streaming client for OpenAI-compatible Chat Completions endpoints.

Covers Ollama (``/v1``), the llama.cpp server, vLLM and hosted providers that
expose the same API, over plain ``httpx``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx

from codebase_qa.answer.backend import BackendError, BackendUnavailableError

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "qwen2.5-coder:7b"
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


class OpenAICompatibleBackend:
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = 0.0,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def request_body(self, system: str, user: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        return body

    def stream(self, system: str, user: str) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        try:
            with self._client.stream("POST", url, json=self.request_body(system, user)) as response:
                if response.status_code >= 400:
                    response.read()
                    message = f"{url} returned HTTP {response.status_code}: "
                    message += _error_message(response)
                    if response.status_code == 404:
                        message += f" (with Ollama, run `ollama pull {self.model}`)"
                    raise BackendError(message)
                yield from _strip_think(_sse_text(response.iter_lines()))
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise BackendUnavailableError(f"cannot reach {self.base_url}: {error}") from error
        except httpx.TimeoutException as error:
            raise BackendError(f"{url} timed out: {error}") from error
        except httpx.HTTPError as error:
            raise BackendError(f"request to {url} failed: {error}") from error


def _sse_text(lines: Iterator[str]) -> Iterator[str]:
    """Content deltas from a Chat Completions event stream.

    Comment lines (``: keep-alive``), chunks without choices (usage reports)
    and reasoning fields are skipped; an ``error`` object mid-stream raises.
    """
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            return
        try:
            event: Any = json.loads(data)
        except json.JSONDecodeError as error:
            raise BackendError(f"malformed stream event: {data[:200]}") from error
        if _field(event, "error") is not None:
            raise BackendError(f"stream error: {_describe(_field(event, 'error'))}")
        choices: list[Any] = _field(event, "choices") or []
        for choice in choices:
            content = _field(_field(choice, "delta"), "content")
            if isinstance(content, str) and content:
                yield content


def _strip_think(pieces: Iterator[str]) -> Iterator[str]:
    """Drops ``<think>...</think>`` spans that some reasoning models put in content."""
    pending, thinking = "", False
    for piece in pieces:
        pending += piece
        while pending:
            tag = _THINK_CLOSE if thinking else _THINK_OPEN
            at = pending.find(tag)
            if at >= 0:
                if not thinking and pending[:at]:
                    yield pending[:at]
                pending, thinking = pending[at + len(tag) :], not thinking
                continue
            keep = _partial_suffix(pending, tag)
            if not thinking and len(pending) > keep:
                yield pending[: len(pending) - keep]
            pending = pending[len(pending) - keep :]
            break
    if pending and not thinking:
        yield pending


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


def _error_message(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    error = _field(body, "error")
    return _describe(error) if error is not None else response.text[:500]


def _describe(error: Any) -> str:
    message = _field(error, "message")
    return str(message) if message else str(error)


def _field(obj: Any, key: str) -> Any:
    """``obj[key]`` when ``obj`` is a JSON object, else None."""
    return cast(dict[str, Any], obj).get(key) if isinstance(obj, dict) else None
