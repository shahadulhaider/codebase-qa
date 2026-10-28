import csv
from collections.abc import Sequence
from pathlib import Path

from conftest import GitRepoFactory, git
from typer.testing import CliRunner

from codebase_qa.chunking import whitespace_token_count
from codebase_qa.cli import app
from codebase_qa.embeddings import Embedder, HashEmbedder
from codebase_qa.eval.report import write_report
from codebase_qa.eval.runner import ModelSpec, load_config, run_stage

FILES = {
    "pkg/timeouts.py": (
        "class Timeout:\n"
        '    """Connect, read, write and pool timeouts in seconds."""\n'
        "\n"
        "    def __init__(self, connect, read):\n"
        "        self.connect = connect\n"
        "        self.read = read\n"
    ),
    "pkg/redirects.py": (
        "def follow_redirect(response, max_redirects):\n"
        "    if max_redirects <= 0:\n"
        "        raise TooManyRedirects(response)\n"
        "    return response.next_request\n"
    ),
    "README.md": "# pkg\n\nAn example package with timeouts and redirects.\n",
}


class LengthReranker:
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [-float(len(t)) for t in texts]


def write_stage(tmp_path: Path, repo: Path, commit: str) -> Path:
    (tmp_path / "questions").mkdir()
    (tmp_path / "questions" / "pkg.yaml").write_text(
        f"""\
repo: {repo}
ref: main
commit: {commit}
questions:
  - id: pkg-1
    question: Timeout connect read seconds
    type: identifier
    gold:
      - {{file: pkg/timeouts.py, symbol: Timeout}}
  - id: pkg-2
    question: what happens when there are too many redirects
    type: conceptual
    gold:
      - {{file: pkg/redirects.py, symbol: follow_redirect}}
      - {{file: README.md}}
"""
    )
    stage = tmp_path / "stages" / "smoke.yaml"
    stage.parent.mkdir()
    stage.write_text(
        """\
name: smoke
questions: [../questions/pkg.yaml]
models:
  # An Ollama spec whose tag matches HashEmbedder's model id, so the
  # injected test embedder owns the build.
  - {name: hash, ollama: hash-64}
chunkers: [treesitter, window]
modes: [dense, bm25, hybrid]
rerank: [false, true]
ks: [1, 3]
"""
    )
    return stage


def test_stage_runs_caches_indexes_and_writes_reports(
    tmp_path: Path, make_repo: GitRepoFactory
) -> None:
    repo = make_repo(FILES)
    commit = git(repo, "rev-parse", "HEAD").strip()
    config = load_config(write_stage(tmp_path, repo, commit))
    assert config.ks == (1, 3)
    assert config.models == (ModelSpec("hash", ollama="hash-64"),)

    embedders: list[HashEmbedder] = []

    def load(spec: ModelSpec) -> Embedder:
        embedders.append(HashEmbedder())
        return embedders[-1]

    def run(log: list[str]):
        return run_stage(
            config,
            tmp_path / "cache",
            log=log.append,
            load_embedder=load,
            token_count=whitespace_token_count,
            reranker=LengthReranker,
        )

    first_log: list[str] = []
    results = run(first_log)

    # 2 chunkers x (dense, hybrid) x rerank, plus BM25 once per chunker x rerank.
    runs = {o.run for o in results.outcomes}
    assert len(runs) == 2 * 2 * 2 + 2 * 2
    assert {r.model for r in runs if r.mode == "bm25"} == {"-"}
    assert len(results.outcomes) == len(runs) * 2
    assert sum("indexing" in line for line in first_log) == 2
    assert [(b.chunker, b.truncated) for b in results.builds] == [
        ("treesitter", None),
        ("window", None),
    ]

    bm25 = next(
        o
        for o in results.outcomes
        if o.run.mode == "bm25" and o.run.chunker == "treesitter" and not o.run.rerank
    )
    assert bm25.question.id == "pkg-1"
    assert min(bm25.ranks["symbol"].values()) == 1

    second_log: list[str] = []
    run(second_log)
    assert not any("indexing" in line for line in second_log)
    assert embedders[-1].documents_embedded == 0

    paths = write_report(results, tmp_path / "results", tmp_path / "charts", "stamp")
    names = sorted(p.name for p in paths)
    assert names == [
        "smoke-by-type.png",
        "smoke-recall.png",
        "stamp.csv",
        "stamp.md",
        "stamp.questions.csv",
    ]
    markdown = (tmp_path / "results" / "stamp.md").read_text()
    assert "## pkg (main" in markdown
    assert "| treesitter · bm25 |" in markdown
    assert "identifier (1)" in markdown
    assert "](../charts/smoke-recall.png)" in markdown

    with (tmp_path / "results" / "stamp.csv").open() as f:
        rows = list(csv.DictReader(f))
    metrics = {r["metric"] for r in rows}
    assert metrics == {"recall@1", "recall@3", "mrr@3", "ms_per_query"}
    with (tmp_path / "results" / "stamp.questions.csv").open() as f:
        questions = list(csv.DictReader(f))
    assert len(questions) == len(results.outcomes)
    assert {q["symbol_found"] for q in questions if q["id"] == "pkg-2"} <= {"0/2", "1/2", "2/2"}


def test_eval_check_cli_reports_counts_and_failures(
    tmp_path: Path, make_repo: GitRepoFactory
) -> None:
    repo = make_repo(FILES)
    commit = git(repo, "rev-parse", "HEAD").strip()
    write_stage(tmp_path, repo, commit)
    questions = tmp_path / "questions" / "pkg.yaml"
    cache = ["--cache-dir", str(tmp_path / "cache")]

    ok = CliRunner().invoke(app, ["eval", "check", str(questions), *cache])
    assert ok.exit_code == 0, ok.output
    assert "pkg: 2 questions (identifier 1, conceptual 1, cross-file 0)" in ok.output

    questions.write_text(questions.read_text().replace("symbol: Timeout}", "symbol: Deadline}"))
    bad = CliRunner().invoke(app, ["eval", "check", str(questions), *cache])
    assert bad.exit_code == 1
    assert "no symbol Deadline in pkg/timeouts.py" in bad.output
