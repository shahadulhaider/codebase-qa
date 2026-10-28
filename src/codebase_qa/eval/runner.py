"""Runs a stage of the retrieval eval: index each build once, then score every question."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

from codebase_qa.chunking import ChunkerName, TokenCounter
from codebase_qa.embeddings import (
    DEFAULT_OLLAMA_URL,
    PRESETS,
    Embedder,
    EmbedderPreset,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    load_token_counter,
    resolve_preset,
)
from codebase_qa.eval.dataset import Question, QuestionSet, load_questions, resolve_spans
from codebase_qa.eval.metrics import (
    LEVELS,
    GoldTarget,
    Level,
    Spans,
    first_hit_ranks,
    level_targets,
)
from codebase_qa.git import GitError, GitRepo
from codebase_qa.indexer import CATALOG_FILE, IndexOptions, index_repo, is_up_to_date
from codebase_qa.retrieval import Mode, RetrievalConfig, Retriever
from codebase_qa.retrieval.rerank import Reranker
from codebase_qa.store import Store

RESULT_DEPTH = 50
"""Chunks retrieved per question; deduplication to targets needs more than max(k)."""

Log = Callable[[str], None]


class EvalConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    """An embedder in the sweep: a sentence-transformers preset or an Ollama model."""

    name: str
    preset: str | None = None
    ollama: str | None = None
    base_url: str = DEFAULT_OLLAMA_URL
    revision: str | None = None
    query_prefix: str = ""
    max_seq_length: int = 2048
    tokenizer: str | None = None
    """Hugging Face tokenizer id that counts tokens for an Ollama model's truncation rate."""

    @property
    def model_id(self) -> str:
        if self.ollama:
            return self.ollama
        return resolve_preset(self.preset or self.name).model_id

    def load(self, device: str | None = None) -> Embedder:
        if self.ollama:
            return OllamaEmbedder(
                self.ollama,
                base_url=self.base_url,
                revision=self.revision,
                query_prefix=self.query_prefix,
                max_seq_length=self.max_seq_length,
            )
        return SentenceTransformerEmbedder(resolve_preset(self.preset or self.name), device=device)

    def token_counter(self, embedder: Embedder) -> Callable[[Sequence[str]], list[int]] | None:
        if isinstance(embedder, SentenceTransformerEmbedder):
            return embedder.count_tokens
        if self.tokenizer:
            count = load_token_counter(EmbedderPreset(self.tokenizer, self.tokenizer, None))
            return lambda texts: [count(t) for t in texts]
        return None


@dataclass(frozen=True)
class StageConfig:
    name: str
    question_files: tuple[Path, ...]
    models: tuple[ModelSpec, ...]
    chunkers: tuple[ChunkerName, ...] = ("treesitter",)
    modes: tuple[Mode, ...] = ("dense",)
    rerank: tuple[bool, ...] = (False,)
    ks: tuple[int, ...] = (1, 5, 10)


def load_config(path: Path) -> StageConfig:
    """Parse a stage file; question set paths are relative to the stage file."""
    import yaml

    raw = cast(dict[str, Any], yaml.safe_load(path.read_text()))
    try:
        models = tuple(
            ModelSpec(**cast(dict[str, Any], m)) if isinstance(m, dict) else ModelSpec(str(m))
            for m in raw["models"]
        )
        config = StageConfig(
            name=str(raw.get("name", path.stem)),
            question_files=tuple((path.parent / q).resolve() for q in raw["questions"]),
            models=models,
            chunkers=tuple(raw.get("chunkers", ["treesitter"])),
            modes=tuple(raw.get("modes", ["dense"])),
            rerank=tuple(bool(r) for r in raw.get("rerank", [False])),
            ks=tuple(int(k) for k in raw.get("ks", [1, 5, 10])),
        )
    except (KeyError, TypeError) as error:
        raise EvalConfigError(f"{path}: missing or malformed field {error}") from error
    for model in config.models:
        if not model.ollama and (model.preset or model.name) not in PRESETS:
            raise EvalConfigError(f"{path}: model {model.name!r} is not a built-in preset")
    for chunker in config.chunkers:
        if chunker not in ("treesitter", "window"):
            raise EvalConfigError(f"{path}: unknown chunker {chunker!r}")
    for mode in config.modes:
        if mode not in ("dense", "bm25", "hybrid"):
            raise EvalConfigError(f"{path}: unknown mode {mode!r}")
    return config


@dataclass(frozen=True)
class RunKey:
    corpus: str
    model: str
    """``-`` for BM25, which does not depend on the embedder."""
    chunker: ChunkerName
    mode: Mode
    rerank: bool

    @property
    def label(self) -> str:
        parts = (
            [self.chunker, self.mode]
            if self.model == "-"
            else [self.model, self.chunker, self.mode]
        )
        return " · ".join(parts) + (" + rerank" if self.rerank else "")


@dataclass(frozen=True)
class QuestionOutcome:
    run: RunKey
    question: Question
    ranks: dict[Level, dict[GoldTarget, int]]
    n_gold: dict[Level, int]
    top: tuple[str, ...]
    """Locations of the first few distinct results, for inspecting misses."""
    seconds: float


@dataclass(frozen=True)
class BuildInfo:
    corpus: str
    model: str
    chunker: ChunkerName
    chunks: int
    truncated: float | None
    """Share of chunks longer than the model's sequence limit; None when not measurable."""


@dataclass
class EvalResults:
    config: StageConfig
    corpora: list[QuestionSet] = field(default_factory=lambda: list[QuestionSet]())
    outcomes: list[QuestionOutcome] = field(default_factory=lambda: list[QuestionOutcome]())
    builds: list[BuildInfo] = field(default_factory=lambda: list[BuildInfo]())


def ensure_clone(url: str, commit: str, dest: Path) -> GitRepo:
    """A clone of ``url`` at ``dest`` that contains ``commit``.

    The clone has no checkout: indexing reads the commit's tree, not a working tree.
    """
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--no-checkout", url, str(dest))
    repo = GitRepo(dest)
    try:
        repo.rev_parse(commit)
    except GitError:
        _git("-C", str(dest), "fetch", "--quiet", "origin", commit)
        repo.rev_parse(commit)
    return repo


def run_stage(
    config: StageConfig,
    cache_dir: Path,
    *,
    device: str | None = None,
    log: Log = print,
    load_embedder: Callable[[ModelSpec], Embedder] | None = None,
    token_count: TokenCounter | None = None,
    reranker: Callable[[], Reranker] | None = None,
) -> EvalResults:
    """Index every (model, chunker) build of every corpus, then score each question.

    Indexes are cached under ``cache_dir/<corpus>/`` and reused when already at
    the pinned commit, so a later stage only pays for queries. BM25 runs once
    per chunker, because its ranking does not depend on the embedder.
    """

    def load(spec: ModelSpec) -> Embedder:
        return load_embedder(spec) if load_embedder else spec.load(device)

    if token_count is None:
        token_count = load_token_counter(PRESETS["minilm"])
    results = EvalResults(config)
    shared_reranker = _once(reranker) if reranker else _default_reranker(device)

    for path in config.question_files:
        questions = load_questions(path)
        results.corpora.append(questions)
        log(f"{questions.name}: {len(questions.questions)} questions at {questions.ref}")
        repo = ensure_clone(
            questions.repo_url, questions.commit, cache_dir / questions.name / "repo"
        )
        spans = resolve_spans(repo, questions)
        index_dir = cache_dir / questions.name / "index"
        bm25_done: set[ChunkerName] = set()

        for spec in config.models:
            embedder = _once(partial(load, spec))
            for chunker in config.chunkers:
                options = IndexOptions(chunker=chunker)
                if not is_up_to_date(
                    repo.root,
                    index_dir,
                    spec.model_id,
                    spec.revision,
                    options,
                    rev=questions.commit,
                ):
                    log(f"  indexing {spec.name} · {chunker}")
                    report = index_repo(
                        repo.root, index_dir, embedder(), token_count, options, rev=questions.commit
                    )
                    log(
                        f"    {report.chunks} chunks, embedded {report.sync.embedded}, "
                        f"reused {report.sync.reused}"
                    )
                runs: list[tuple[Mode, bool]] = [
                    (mode, rerank)
                    for mode in config.modes
                    for rerank in config.rerank
                    if mode != "bm25" or chunker not in bm25_done
                ]
                if "bm25" in config.modes:
                    bm25_done.add(chunker)
                if any(mode != "bm25" for mode, _ in runs):
                    results.builds.append(
                        _build_info(questions.name, spec, chunker, index_dir, embedder())
                    )
                with Retriever.open(
                    index_dir,
                    model_id=spec.model_id,
                    chunker=chunker,
                    embedder=embedder,
                    reranker=shared_reranker,
                ) as retriever:
                    for mode, rerank in runs:
                        key = RunKey(
                            questions.name,
                            "-" if mode == "bm25" else spec.name,
                            chunker,
                            mode,
                            rerank,
                        )
                        log(f"  {key.label}")
                        results.outcomes += [
                            _score(retriever, key, q, spans) for q in questions.questions
                        ]
    return results


def _score(
    retriever: Retriever, key: RunKey, question: Question, spans: dict[GoldTarget, Spans]
) -> QuestionOutcome:
    config = RetrievalConfig(mode=key.mode, k=RESULT_DEPTH, rerank=key.rerank)
    started = time.perf_counter()
    found = retriever.retrieve(question.text, config)
    seconds = time.perf_counter() - started
    chunks = [r.chunk for r in found]
    ranks: dict[Level, dict[GoldTarget, int]] = {
        level: first_hit_ranks(chunks, question.gold, spans, level) for level in LEVELS
    }
    n_gold: dict[Level, int] = {level: len(level_targets(question.gold, level)) for level in LEVELS}
    top = tuple(dict.fromkeys(r.location for r in found))[:5]
    return QuestionOutcome(key, question, ranks, n_gold, top, seconds)


def _build_info(
    corpus: str, spec: ModelSpec, chunker: ChunkerName, index_dir: Path, embedder: Embedder
) -> BuildInfo:
    store = Store.open(index_dir / CATALOG_FILE)
    try:
        texts = [c.embedding_text for c in store.chunks(chunker)]
    finally:
        store.close()
    counter = spec.token_counter(embedder)
    truncated = None
    if counter is not None and texts:
        limit = embedder.max_seq_length
        truncated = sum(1 for n in counter(texts) if n > limit) / len(texts)
    return BuildInfo(corpus, spec.name, chunker, len(texts), truncated)


def _once[T](factory: Callable[[], T]) -> Callable[[], T]:
    cache: list[T] = []

    def get() -> T:
        if not cache:
            cache.append(factory())
        return cache[0]

    return get


def _default_reranker(device: str | None) -> Callable[[], Reranker]:
    def load() -> Reranker:
        from codebase_qa.retrieval import CrossEncoderReranker

        return CrossEncoderReranker(device=device)

    return _once(load)


def _git(*args: str) -> None:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or f"git {args[0]} failed")
