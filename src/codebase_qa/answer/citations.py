"""Validate ``[n]`` citations in a streamed answer against the numbered sources."""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_PARTIAL = re.compile(r"\[[\d\s,]*")
_MAX_PARTIAL = 32
"""Longest unfinished ``[1, 2, ...`` held back before it is emitted as plain text."""


class CitationFilter:
    """Streams text through, dropping citations to source numbers that don't exist.

    Text inside Markdown code spans and fences is passed through untouched, so
    ``items[0]`` in a code block is never read as a citation. Possible citation
    prefixes and backtick runs split across chunks are held back until the next
    :meth:`feed` or :meth:`flush` resolves them.
    """

    def __init__(self, source_count: int) -> None:
        self.source_count = source_count
        self.cited: list[int] = []
        """Valid source numbers in order of first citation."""
        self.invalid: list[int] = []
        """Cited numbers with no matching source, in order of first citation."""
        self._pending = ""
        self._fence: str | None = None

    def feed(self, text: str) -> str:
        self._pending += text
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        buf, out, i = self._pending, list[str](), 0
        while i < len(buf):
            ch = buf[i]
            if ch == "`":
                end = i
                while end < len(buf) and buf[end] == "`":
                    end += 1
                if end == len(buf) and not final:
                    break
                self._toggle_code(buf[i:end])
                out.append(buf[i:end])
                i = end
            elif ch == "[" and self._fence is None:
                match = _CITATION.match(buf, i)
                if match:
                    self._emit_citation(match.group(1), out)
                    i = match.end()
                    continue
                partial = _PARTIAL.match(buf, i)
                assert partial is not None
                if not final and partial.end() == len(buf) and len(partial.group()) < _MAX_PARTIAL:
                    if out and out[-1] == " ":
                        out.pop()
                        i -= 1
                    break
                out.append(ch)
                i += 1
            elif ch == " " and i == len(buf) - 1 and not final:
                break  # a dropped citation that follows also drops this space
            else:
                out.append(ch)
                i += 1
        self._pending = buf[i:]
        return "".join(out)

    def _toggle_code(self, run: str) -> None:
        if self._fence is None:
            self._fence = run
        elif run == self._fence or (len(self._fence) >= 3 and len(run) >= len(self._fence)):
            self._fence = None

    def _emit_citation(self, body: str, out: list[str]) -> None:
        valid: list[int] = []
        for number in (int(part) for part in body.split(",")):
            if 1 <= number <= self.source_count:
                valid.append(number)
                if number not in self.cited:
                    self.cited.append(number)
            elif number not in self.invalid:
                self.invalid.append(number)
        if valid:
            out.append("[" + ", ".join(map(str, valid)) + "]")
        elif out and out[-1] == " ":
            out.pop()


def check_citations(text: str, source_count: int) -> tuple[str, CitationFilter]:
    """Non-streaming form: the cleaned text and the filter holding the citation lists."""
    citations = CitationFilter(source_count)
    return citations.feed(text) + citations.flush(), citations
