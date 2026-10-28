import json
from collections.abc import Callable

import httpx
import pytest

from codebase_qa.answer import BackendError, BackendUnavailableError, OpenAICompatibleBackend

Handler = Callable[[httpx.Request], httpx.Response]


def sse(*events: object, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def delta(text: str | None, **extra: object) -> dict[str, object]:
    return {"choices": [{"index": 0, "delta": {"content": text, **extra}}]}


def backend(handler: Handler, **kwargs: object) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        "http://llm.test/v1/",
        "coder",
        transport=httpx.MockTransport(handler),
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


def test_request_shape_and_streamed_content() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=sse(delta("Hello"), delta(" world")))

    llm = backend(handler, api_key="secret", max_tokens=512)
    assert "".join(llm.stream("be brief", "question")) == "Hello world"

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    assert json.loads(request.content) == {
        "model": "coder",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "question"},
        ],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": 512,
    }


def test_no_authorization_header_without_a_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=sse(delta("ok")))

    list(backend(handler).stream("s", "u"))
    assert "authorization" not in seen[0].headers


def test_keep_alives_usage_chunks_reasoning_and_null_content_are_skipped() -> None:
    body = b": OPENROUTER PROCESSING\n\n" + sse(
        delta(None, role="assistant"),
        delta("", reasoning="thinking..."),
        delta(None, reasoning_content="more thinking"),
        delta("answer"),
        {"choices": [], "usage": {"prompt_tokens": 3}},
    )
    llm = backend(lambda _: httpx.Response(200, content=body))
    assert list(llm.stream("s", "u")) == ["answer"]


def test_think_spans_in_content_are_removed_even_when_split() -> None:
    pieces = ["<thi", "nk>plan the ans", "wer</th", "ink>The ", "answer"]
    llm = backend(lambda _: httpx.Response(200, content=sse(*(delta(p) for p in pieces))))
    assert "".join(llm.stream("s", "u")) == "The answer"


def test_http_error_reports_the_provider_message() -> None:
    body = {"error": {"message": "invalid api key", "code": 401}}
    llm = backend(lambda _: httpx.Response(401, json=body))
    with pytest.raises(BackendError, match="HTTP 401: invalid api key"):
        list(llm.stream("s", "u"))


def test_missing_model_suggests_pulling_it() -> None:
    body = {"error": {"message": 'model "coder" not found', "type": "not_found_error"}}
    llm = backend(lambda _: httpx.Response(404, json=body))
    with pytest.raises(BackendError, match="ollama pull coder"):
        list(llm.stream("s", "u"))


def test_error_event_mid_stream_raises_after_the_text_so_far() -> None:
    body = sse(delta("partial"), {"error": {"message": "upstream overloaded"}}, done=False)
    received: list[str] = []
    with pytest.raises(BackendError, match="upstream overloaded"):
        for piece in backend(lambda _: httpx.Response(200, content=body)).stream("s", "u"):
            received.append(piece)
    assert received == ["partial"]


def test_unreachable_server_is_reported_as_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(BackendUnavailableError, match=r"cannot reach http://llm\.test/v1"):
        list(backend(handler).stream("s", "u"))
