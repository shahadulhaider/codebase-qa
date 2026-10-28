from collections.abc import Iterator

from codebase_qa.answer import AnswerStream
from codebase_qa.chunking import Chunk
from codebase_qa.retrieval import SearchResult


class ScriptedBackend:
    name = "scripted"
    model = "fake"

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.prompts: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.prompts.append((system, user))
        yield from self.pieces


def result(path: str, start: int, end: int, symbol: str = "") -> SearchResult:
    chunk = Chunk(
        file_path=path,
        language="python",
        text="def f():\n    pass",
        start_line=start,
        end_line=end,
        symbol_kind="function" if symbol else "module",
        chunker="treesitter",
        symbol_name=symbol,
        symbol_path=symbol,
    )
    return SearchResult(chunk=chunk, score=1.0)


def test_sources_are_numbered_in_the_prompt_before_the_question() -> None:
    backend = ScriptedBackend(["ok"])
    answer = AnswerStream("where?", [result("a.py", 1, 2, "f"), result("b/c.py", 10, 20)], backend)
    list(answer)

    _, user = backend.prompts[0]
    assert user.index("[1] a.py:1-2 f (function)") < user.index("[2] b/c.py:10-20\n")
    assert user.endswith("Question: where?")
    assert "```python\ndef f():\n    pass\n```" in user


def test_streamed_answer_drops_unknown_citations_and_records_cited_sources() -> None:
    backend = ScriptedBackend(["It is in f [", "1]. Also [", "5]. And [2", "]."])
    answer = AnswerStream("q", [result("a.py", 1, 2), result("b.py", 3, 4)], backend)

    assert "".join(answer) == "It is in f [1]. Also. And [2]."
    assert answer.text == "It is in f [1]. Also. And [2]."
    assert answer.cited == [1, 2]
    assert answer.invalid == [5]
