"""Command-line entry point."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from codebase_qa import __version__
from codebase_qa.chunking import ChunkerName, LanguageRegistry, chunk_file, route_file
from codebase_qa.embeddings import DEFAULT_PRESET, PRESETS, load_token_counter

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
