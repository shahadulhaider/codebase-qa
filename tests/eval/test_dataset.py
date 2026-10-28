from pathlib import Path

import pytest
from conftest import GitRepoFactory, git

from codebase_qa.eval.dataset import DatasetError, load_questions, resolve_spans
from codebase_qa.eval.metrics import GoldTarget
from codebase_qa.git import GitRepo

SOURCE = '''\
import os


class Client:
    """A client."""

    def send(self, request):
        return request

    def close(self):
        pass


def helper():
    return os.sep
'''


def write_set(tmp_path: Path, commit: str, gold: str, *, qtype: str = "conceptual") -> Path:
    path = tmp_path / "corpus.yaml"
    path.write_text(
        f"""\
repo: https://example.com/corpus.git
ref: v1
commit: {commit}
questions:
  - id: q1
    question: >
      How are requests sent?
    type: {qtype}
    gold:
{gold}
"""
    )
    return path


def test_load_and_resolve_class_method_and_file_spans(
    tmp_path: Path, make_repo: GitRepoFactory
) -> None:
    repo = make_repo({"pkg/client.py": SOURCE, "README.md": "# Corpus\n"})
    commit = git(repo, "rev-parse", "HEAD").strip()
    gold = (
        "      - {file: pkg/client.py, symbol: Client}\n"
        "      - {file: pkg/client.py, symbol: Client.send}\n"
        "      - {file: pkg/client.py, symbol: helper}\n"
        "      - {file: README.md}\n"
    )
    qs = load_questions(write_set(tmp_path, commit, gold))

    assert qs.name == "corpus"
    (q,) = qs.questions
    assert q.text == "How are requests sent?"
    assert q.type == "conceptual"
    spans = resolve_spans(GitRepo(repo), qs)
    assert spans == {
        GoldTarget("pkg/client.py", "Client"): ((4, 11),),
        GoldTarget("pkg/client.py", "Client.send"): ((7, 8),),
        GoldTarget("pkg/client.py", "helper"): ((14, 15),),
    }


def test_separate_declarations_of_one_name_stay_separate_spans(
    tmp_path: Path, make_repo: GitRepoFactory
) -> None:
    source = (
        "export interface Make {\n  id: string;\n}\n"
        + "\n" * 10
        + "export function Make(id: string): Make {\n  return { id };\n}\n"
    )
    repo = make_repo({"core.ts": source})
    commit = git(repo, "rev-parse", "HEAD").strip()
    qs = load_questions(write_set(tmp_path, commit, "      - {file: core.ts, symbol: Make}\n"))
    assert resolve_spans(GitRepo(repo), qs) == {GoldTarget("core.ts", "Make"): ((1, 3), (14, 16))}


def test_all_resolution_problems_are_reported_together(
    tmp_path: Path, make_repo: GitRepoFactory
) -> None:
    repo = make_repo({"pkg/client.py": SOURCE, "README.md": "# Corpus\n"})
    commit = git(repo, "rev-parse", "HEAD").strip()
    gold = (
        "      - {file: pkg/missing.py}\n"
        "      - {file: pkg/client.py, symbol: Client.recv}\n"
        "      - {file: README.md, symbol: Intro}\n"
    )
    qs = load_questions(write_set(tmp_path, commit, gold))
    with pytest.raises(DatasetError) as error:
        resolve_spans(GitRepo(repo), qs)
    message = str(error.value)
    assert "no file pkg/missing.py" in message
    assert "no symbol Client.recv in pkg/client.py" in message
    assert "README.md is not a symbol-chunked file" in message


def test_invalid_type_and_short_commit_are_rejected(tmp_path: Path) -> None:
    gold = "      - {file: a.py}\n"
    with pytest.raises(DatasetError, match="type must be one of"):
        load_questions(write_set(tmp_path, "0" * 40, gold, qtype="vibes"))
    with pytest.raises(DatasetError, match="full 40-character SHA"):
        load_questions(write_set(tmp_path, "abc123", gold))


def test_missing_field_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("repo: x\nref: y\ncommit: z\nquestions:\n  - id: q1\n    type: conceptual\n")
    with pytest.raises(DatasetError, match="missing or malformed field 'gold'"):
        load_questions(path)
