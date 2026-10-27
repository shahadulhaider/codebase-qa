# codebase-qa

Index a Git repository and answer questions about it with cited sources.

`codebase-qa` splits source files into syntax-aware chunks (functions, classes,
methods), embeds them with a local model, and stores them in a vector index.
Questions are answered by hybrid retrieval (BM25 + dense vectors) with optional
reranking, and every answer cites `file:line` ranges from the indexed commit.

## Status

Early development. Interfaces are unstable.

## Development

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```
