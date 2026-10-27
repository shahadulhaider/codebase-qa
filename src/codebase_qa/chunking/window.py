"""Token-budgeted line windows, used for non-code files and as a baseline."""

from __future__ import annotations

from codebase_qa.chunking.base import Chunk, ChunkerName, Line, TokenCounter, split_to_budget


class LineWindowChunker:
    name: ChunkerName = "window"

    def __init__(
        self, token_count: TokenCounter, *, max_tokens: int = 256, overlap_lines: int = 2
    ) -> None:
        self._token_count = token_count
        self._max_tokens = max_tokens
        self._overlap_lines = overlap_lines

    def chunk(self, file_path: str, language: str, source: str) -> list[Chunk]:
        lines = [Line(i + 1, text.rstrip("\r")) for i, text in enumerate(source.split("\n"))]
        while lines and not lines[-1].text.strip():
            lines.pop()
        if not any(line.text.strip() for line in lines):
            return []
        header = Chunk(file_path, language, "", 1, 1, "window", self.name).header
        windows = split_to_budget(
            lines,
            header=header,
            token_count=self._token_count,
            max_tokens=self._max_tokens,
            overlap_lines=self._overlap_lines,
        )
        return [
            Chunk(
                file_path=file_path,
                language=language,
                text="\n".join(line.text for line in window),
                start_line=window[0].number,
                end_line=window[-1].number,
                symbol_kind="window",
                chunker=self.name,
            )
            for window in windows
            if any(line.text.strip() for line in window)
        ]
