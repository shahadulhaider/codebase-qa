"""Command-line entry point."""

import os
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.table import Table

from codebase_qa import __version__
from codebase_qa.chunking import ChunkerName, LanguageRegistry, chunk_file, route_file
from codebase_qa.embeddings import (
    DEFAULT_PRESET,
    PRESETS,
    SentenceTransformerEmbedder,
    load_token_counter,
    preset_for,
    resolve_preset,
)
from codebase_qa.git import GitError
from codebase_qa.indexer import (
    CATALOG_FILE,
    DEFAULT_INDEX_DIR,
    VECTORS_DIR,
    IndexOptions,
    index_repo,
    index_stats,
    is_up_to_date,
)
from codebase_qa.retrieval import (
    CrossEncoderReranker,
    Filters,
    IndexNotFoundError,
    Mode,
    RetrievalConfig,
    Retriever,
    SearchResult,
    StaleIndexError,
)
from codebase_qa.store import SchemaVersionError

REFERENCE_TOKENIZER = "minilm"

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Index a Git repository and answer questions about it."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def chunk(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    chunker: Annotated[ChunkerName, typer.Option(help="Chunking strategy.")] = "treesitter",
    max_tokens: Annotated[int, typer.Option(help="Token budget per chunk.")] = 256,
    model: Annotated[str, typer.Option(help="Preset whose tokenizer counts tokens.")] = (
        DEFAULT_PRESET
    ),
    show: Annotated[bool, typer.Option(help="Print each chunk's text.")] = False,
) -> None:
    """Show how a file is split into chunks."""
    source = file.read_text(errors="replace")
    route = route_file(file.as_posix(), len(source.encode()), registry=LanguageRegistry.default())
    if route is None:
        raise typer.BadParameter(f"{file} is not an indexed file type", param_hint="FILE")
    if model not in PRESETS:
        raise typer.BadParameter(f"choose one of: {', '.join(PRESETS)}", param_hint="--model")
    count = load_token_counter(PRESETS[model])
    chunks = chunk_file(
        file.as_posix(), source, route, chunker=chunker, token_count=count, max_tokens=max_tokens
    )

    console = Console()
    table = Table(title=f"{file} ({route.language}, {chunker}, budget {max_tokens})")
    for column in ("#", "kind", "symbol", "lines", "tokens"):
        table.add_column(column, justify="right" if column in {"#", "tokens"} else "left")
    over = 0
    for i, c in enumerate(chunks, 1):
        tokens = count(c.embedding_text)
        over += tokens > max_tokens
        style = "red" if tokens > max_tokens else None
        table.add_row(
            str(i),
            c.symbol_kind,
            c.symbol_path or "-",
            f"{c.start_line}-{c.end_line}",
            str(tokens),
            style=style,
        )
    console.print(table)
    console.print(f"{len(chunks)} chunks, {over} over budget")
    if show:
        for i, c in enumerate(chunks, 1):
            console.rule(f"#{i} {c.symbol_path or c.symbol_kind} {c.start_line}-{c.end_line}")
            console.print(c.embedding_text, markup=False, highlight=False)


IndexDirOption = Annotated[
    Path | None,
    typer.Option(
        envvar="CODEBASE_QA_INDEX_DIR",
        help=f"Index location [default: <repo>/{DEFAULT_INDEX_DIR}].",
    ),
]


@app.command()
def index(
    repo: Annotated[Path, typer.Argument(exists=True, file_okay=False)] = Path("."),
    rev: Annotated[str, typer.Option(help="Commit, branch or tag to index.")] = "HEAD",
    chunker: Annotated[ChunkerName, typer.Option(help="Chunking strategy.")] = "treesitter",
    model: Annotated[
        str,
        typer.Option(envvar="CODEBASE_QA_MODEL", help="Embedder preset or Hugging Face id."),
    ] = DEFAULT_PRESET,
    revision: Annotated[
        str | None, typer.Option(help="Model revision for a Hugging Face id.")
    ] = None,
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom model code for a Hugging Face id.")
    ] = False,
    device: Annotated[
        str | None, typer.Option(envvar="CODEBASE_QA_DEVICE", help="cpu, cuda or mps.")
    ] = None,
    exclude: Annotated[
        list[str] | None, typer.Option(help="Extra path glob to skip; repeatable.")
    ] = None,
    index_dir: IndexDirOption = None,
    rebuild: Annotated[bool, typer.Option(help="Delete the index and start over.")] = False,
    offline: Annotated[bool, typer.Option(help="Use only cached models.")] = False,
    verify: Annotated[
        bool, typer.Option(help="Reconcile vectors with the catalog even if up to date.")
    ] = False,
) -> None:
    """Index a repository revision."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    target = index_dir or repo / DEFAULT_INDEX_DIR
    if rebuild:
        _remove_index(target)
    preset = resolve_preset(model, revision=revision, trust_remote_code=trust_remote_code)
    options = IndexOptions(chunker=chunker, excludes=tuple(exclude or ()), verify=verify)
    console = Console()
    try:
        current = is_up_to_date(repo, target, preset.model_id, preset.revision, options, rev=rev)
    except (GitError, SchemaVersionError) as error:
        raise typer.BadParameter(str(error)) from error
    if current:
        console.print(f"index at {target} is up to date for {preset.name} ({chunker})")
        return
    with console.status(f"loading {preset.model_id}"):
        embedder = SentenceTransformerEmbedder(preset, device=device)
        token_count = load_token_counter(PRESETS[REFERENCE_TOKENIZER])
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("embedding", total=None)

        def on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        try:
            report = index_repo(
                repo, target, embedder, token_count, options, rev=rev, on_progress=on_progress
            )
        except (GitError, SchemaVersionError) as error:
            raise typer.BadParameter(str(error)) from error
    if report.up_to_date:
        console.print(f"{report.collection} is up to date at {report.commit[:12]}")
        return
    c, s = report.changes, report.sync
    console.print(
        f"indexed {report.commit[:12]} into {report.collection}\n"
        f"  files:  {report.files} ({report.skipped} skipped)  "
        f"+{c.added} ~{c.modified} -{c.deleted} ={c.unchanged}\n"
        f"  chunks: {report.chunks}  embedded {s.embedded}, reused {s.reused}, "
        f"kept {s.kept}, removed {s.removed}"
    )


@app.command()
def stats(
    repo: Annotated[Path, typer.Argument(exists=True, file_okay=False)] = Path("."),
    index_dir: IndexDirOption = None,
) -> None:
    """Summarise an index: files, chunks and vector collections."""
    target = index_dir or repo / DEFAULT_INDEX_DIR
    if not (target / CATALOG_FILE).exists():
        raise typer.BadParameter(f"no index at {target}; run `codebase-qa index` first")
    try:
        summary = index_stats(target)
    except SchemaVersionError as error:
        raise typer.BadParameter(str(error)) from error
    console = Console()
    console.print(f"revision {(summary.current_rev or '-')[:12]}, {summary.files} files")
    console.print(
        "chunks: " + ", ".join(f"{name} {count}" for name, count in summary.chunks.items())
    )
    console.print("languages: " + ", ".join(f"{k} {v}" for k, v in summary.languages.items()))
    console.print("kinds: " + ", ".join(f"{k} {v}" for k, v in summary.kinds.items()))
    table = Table(title="collections")
    for column in ("collection", "model", "chunker", "vectors", "indexed", "state"):
        table.add_column(column)
    for status in summary.builds:
        b = status.build
        table.add_row(
            b.collection,
            b.model_id,
            b.chunker,
            str(status.vectors),
            b.indexed_rev[:12],
            "stale" if status.stale else "fresh",
            style="yellow" if status.stale else None,
        )
    console.print(table)


def _remove_index(target: Path) -> None:
    if not target.exists():
        return
    known = {CATALOG_FILE, f"{CATALOG_FILE}-wal", f"{CATALOG_FILE}-shm", VECTORS_DIR}
    unexpected = {p.name for p in target.iterdir()} - known
    if unexpected:
        raise typer.BadParameter(
            f"refusing to delete {target}: it contains {', '.join(sorted(unexpected))}"
        )
    shutil.rmtree(target)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural-language or identifier query.")],
    repo: Annotated[Path, typer.Argument(exists=True, file_okay=False)] = Path("."),
    k: Annotated[int, typer.Option("--k", "-k", help="Number of results.")] = 10,
    mode: Annotated[Mode, typer.Option(help="Retriever.")] = "dense",
    lang: Annotated[list[str] | None, typer.Option(help="Language filter; repeatable.")] = None,
    kind: Annotated[list[str] | None, typer.Option(help="Symbol kind filter; repeatable.")] = None,
    role: Annotated[
        list[str] | None, typer.Option(help="source, test, docs or config; repeatable.")
    ] = None,
    path: Annotated[str | None, typer.Option(help="Path glob, e.g. 'httpx/_*.py'.")] = None,
    chunker: Annotated[ChunkerName | None, typer.Option(help="Build to query.")] = None,
    model: Annotated[
        str | None,
        typer.Option(envvar="CODEBASE_QA_MODEL", help="Embedder preset or Hugging Face id."),
    ] = None,
    device: Annotated[
        str | None, typer.Option(envvar="CODEBASE_QA_DEVICE", help="cpu, cuda or mps.")
    ] = None,
    index_dir: IndexDirOption = None,
    candidate_k: Annotated[
        int, typer.Option(help="Results per retriever before hybrid fusion.")
    ] = RetrievalConfig.candidate_k,
    rrf_k: Annotated[int, typer.Option(help="RRF rank constant.")] = RetrievalConfig.rrf_k,
    rerank: Annotated[bool, typer.Option(help="Rerank with a cross-encoder.")] = False,
    rerank_k: Annotated[
        int, typer.Option(help="Top results the cross-encoder reorders.")
    ] = RetrievalConfig.rerank_k,
    show: Annotated[bool, typer.Option(help="Print each result's text.")] = False,
) -> None:
    """Search an index and print ranked chunks."""
    target = index_dir or repo / DEFAULT_INDEX_DIR
    model_id = resolve_preset(model).model_id if model else None
    console = Console()
    try:
        config = RetrievalConfig(
            mode=mode,
            k=k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            rerank=rerank,
            rerank_k=rerank_k,
            filters=Filters(
                languages=tuple(lang or ()),
                kinds=tuple(kind or ()),
                roles=tuple(role or ()),
                path_glob=path,
            ),
        )
        retriever = Retriever.open(
            target,
            model_id=model_id,
            chunker=chunker,
            embedder=lambda: _load_embedder(console, build.model_id, build.model_revision, device),
            reranker=lambda: _load_reranker(console, device),
        )
    except (ValueError, IndexNotFoundError, StaleIndexError, SchemaVersionError) as error:
        raise typer.BadParameter(str(error)) from error
    build = retriever.searcher.build
    with retriever:
        results = retriever.retrieve(query, config)

    title = f"{mode}{' + rerank' if rerank else ''} · {build.collection} · {build.indexed_rev[:12]}"
    table = Table(title=title)
    columns = _result_columns(config)
    for column in ("#", *columns, "location", "symbol", "kind"):
        numeric = column == "#" or column in columns
        table.add_column(column, justify="right" if numeric else "left", overflow="fold")
    for rank, r in enumerate(results, 1):
        c = r.chunk
        cells = [_result_cell(r, column, config) for column in columns]
        table.add_row(str(rank), *cells, r.location, c.symbol_path or "-", c.symbol_kind)
    console.print(table)
    if not results:
        console.print("no results")
    if show:
        for rank, r in enumerate(results, 1):
            console.rule(f"#{rank} {r.location} {r.chunk.symbol_path}")
            console.print(r.chunk.text, markup=False, highlight=False)


def _load_embedder(
    console: Console, model_id: str, revision: str | None, device: str | None
) -> SentenceTransformerEmbedder:
    with console.status(f"loading {model_id}"):
        return SentenceTransformerEmbedder(preset_for(model_id, revision), device=device)


def _load_reranker(console: Console, device: str | None) -> CrossEncoderReranker:
    with console.status("loading reranker"):
        return CrossEncoderReranker(device=device)


def _result_columns(config: RetrievalConfig) -> list[str]:
    columns = {"dense": ["similarity"], "bm25": ["bm25"], "hybrid": ["rrf", "dense#", "bm25#"]}[
        config.mode
    ]
    return [*columns, "rerank", "moved"] if config.rerank else columns


def _result_cell(result: SearchResult, column: str, config: RetrievalConfig) -> str:
    scores, ranks = result.scores, result.ranks
    match column:
        case "similarity":
            return f"{1.0 - scores['dense_distance']:.3f}"
        case "bm25":
            return f"{scores['bm25']:.3f}"
        case "rrf":
            return f"{scores['rrf']:.4f}"
        case "dense#" | "bm25#":
            stage_rank = ranks.get(column.removesuffix("#"))
            return "-" if stage_rank is None else str(stage_rank)
        case "rerank":
            return f"{scores['rerank']:.2f}" if "rerank" in scores else "-"
        case _:
            if "rerank" not in ranks:
                return "-"
            moved = ranks[config.mode] - ranks["rerank"]
            return f"↑{moved}" if moved > 0 else f"↓{-moved}" if moved < 0 else "="
