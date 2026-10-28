"""Question sets: questions with gold targets, pinned to one commit of one repository."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

from codebase_qa.chunking import LanguageRegistry, chunk_file, route_file, whitespace_token_count
from codebase_qa.eval.metrics import GoldTarget, Span, Spans
from codebase_qa.git import GitRepo

QuestionType = Literal["identifier", "conceptual", "cross-file"]
QUESTION_TYPES: tuple[QuestionType, ...] = get_args(QuestionType)


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    type: QuestionType
    gold: tuple[GoldTarget, ...]


@dataclass(frozen=True)
class QuestionSet:
    name: str
    repo_url: str
    ref: str
    """The release tag or branch the commit was taken from, for readers."""
    commit: str
    questions: tuple[Question, ...]

    @property
    def gold_targets(self) -> set[GoldTarget]:
        return {g for q in self.questions for g in q.gold}


def load_questions(path: Path) -> QuestionSet:
    """Parse a question set file.

    The format is YAML with ``repo``, ``ref``, ``commit`` and a ``questions``
    list; each question has ``id``, ``question``, ``type`` and a ``gold`` list
    of ``{file, symbol?}`` entries.
    """
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise DatasetError(f"{path}: expected a mapping at the top level")
    raw = cast(dict[str, Any], data)
    try:
        questions = tuple(_question(cast(dict[str, Any], q)) for q in raw["questions"])
        qs = QuestionSet(
            name=path.stem,
            repo_url=str(raw["repo"]),
            ref=str(raw["ref"]),
            commit=str(raw["commit"]),
            questions=questions,
        )
    except (KeyError, TypeError) as error:
        raise DatasetError(f"{path}: missing or malformed field {error}") from error
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise DatasetError(f"{path}: duplicate question ids {', '.join(duplicates)}")
    if len(qs.commit) != 40:
        raise DatasetError(f"{path}: commit must be a full 40-character SHA")
    return qs


def resolve_spans(repo: GitRepo, questions: QuestionSet) -> dict[GoldTarget, Spans]:
    """Line spans of every gold symbol at the pinned commit, from the treesitter chunker.

    A symbol's chunks, including its members and the pieces of an oversized
    body, are merged into contiguous spans, so they do not depend on the token
    budget. Separate declarations of one name stay separate spans, so code
    between them is not counted as part of the symbol.
    Every gold file must exist and every gold symbol must resolve; all
    problems are reported together.
    """
    gold = questions.gold_targets
    entries = {e.path: e for e in repo.ls_tree(questions.commit)}
    registry = LanguageRegistry.default()
    problems = sorted(f"no file {g.file}" for g in gold if g.file not in entries)

    wanted = {g.file for g in gold if g.symbol and g.file in entries}
    paths_by_blob: dict[str, list[str]] = defaultdict(list)
    for path in wanted:
        paths_by_blob[entries[path].blob_sha].append(path)
    symbols: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    unsymbolled: set[str] = set()
    for sha, data in repo.read_blobs(paths_by_blob):
        for path in paths_by_blob[sha]:
            route = route_file(path, entries[path].size, registry=registry)
            if route is None or not route.syntax_aware:
                problems.append(f"{path} is not a symbol-chunked file, so it has no symbols")
                unsymbolled.add(path)
                continue
            for c in chunk_file(
                path,
                data.decode("utf-8", errors="replace"),
                route,
                chunker="treesitter",
                token_count=whitespace_token_count,
                registry=registry,
            ):
                if c.symbol_path:
                    symbols[path].append((c.symbol_path, c.start_line, c.end_line))

    spans: dict[GoldTarget, Spans] = {}
    for g in sorted(gold, key=str):
        if not g.symbol or g.file not in wanted or g.file in unsymbolled:
            continue
        lines = [
            (start, end)
            for symbol, start, end in symbols[g.file]
            if symbol == g.symbol or symbol.startswith(g.symbol + ".")
        ]
        if lines:
            spans[g] = _merge(lines)
        else:
            problems.append(f"no symbol {g.symbol} in {g.file}")
    if problems:
        raise DatasetError(f"{questions.name}: " + "; ".join(problems))
    return spans


def _question(raw: dict[str, Any]) -> Question:
    kind = raw["type"]
    if kind not in QUESTION_TYPES:
        raise DatasetError(f"question {raw.get('id')}: type must be one of {QUESTION_TYPES}")
    gold = tuple(
        GoldTarget(str(g["file"]), str(g.get("symbol") or ""))
        for g in cast(list[dict[str, Any]], raw["gold"])
    )
    if not gold:
        raise DatasetError(f"question {raw['id']}: needs at least one gold target")
    return Question(str(raw["id"]), str(raw["question"]).strip(), kind, gold)


MERGE_GAP = 3
"""Lines that may separate two chunks of one definition, e.g. blank lines between methods."""


def _merge(lines: list[Span]) -> Spans:
    merged: list[Span] = []
    for start, end in sorted(lines):
        if merged and start <= merged[-1][1] + MERGE_GAP:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)
