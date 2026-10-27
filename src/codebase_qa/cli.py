"""Command-line entry point."""

import typer

from codebase_qa import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Index a Git repository and answer questions about it."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)
