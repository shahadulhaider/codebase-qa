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
) -> None:
    """Index a repository revision."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    target = index_dir or repo / DEFAULT_INDEX_DIR
    if rebuild:
        _remove_index(target)
    preset = resolve_preset(model, revision=revision, trust_remote_code=trust_remote_code)
    console = Console()
    with console.status(f"loading {preset.model_id}"):
        embedder = SentenceTransformerEmbedder(preset, device=device)
        token_count = load_token_counter(PRESETS[REFERENCE_TOKENIZER])
    options = IndexOptions(chunker=chunker, excludes=tuple(exclude or ()))
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
    console.print(
        f"indexed {report.commit[:12]} into {report.collection}: {report.files} files "
        f"({report.skipped} skipped), {report.chunks} chunks, {report.embedded} embedded"
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
