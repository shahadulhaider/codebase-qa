# Contributor guide

Guidance for anyone changing this repository, human or automated. User-facing docs are in [README.md](README.md). Design and rationale are in [docs/architecture.md](docs/architecture.md).

## Commands

```sh
uv sync --all-extras                    # environment, including optional extras
uv run pytest                           # fast suite; must pass before every commit
uv run pytest -m model                  # real models; run after touching embeddings.py, rerank.py or model presets
uv run ruff check . && uv run ruff format --check .
uv run pyright                          # strict; must report 0 errors
```

Run the narrowest relevant test file while working, then the full fast suite, ruff and pyright before finishing.

## Layout

- `src/codebase_qa/cli.py`: Typer CLI. Flag and env handling and model loading live here, not in the library.
- `src/codebase_qa/chunking/`: file routing, the language registry (`languages.py` + `queries/*.scm`), the chunkers.
- `src/codebase_qa/embeddings.py`: the `Embedder` protocol, presets and backends.
- `src/codebase_qa/store.py`: the SQLite catalog (schema, FTS5, builds).
- `src/codebase_qa/vector_index.py`: Chroma wrapper.
- `src/codebase_qa/indexer.py`: incremental indexing and collection sync.
- `src/codebase_qa/retrieval/`: filters, dense, BM25, fusion, rerank, `Searcher`, `Retriever`.
- `src/codebase_qa/answer/`: prompt, answer backends, citation validation.
- `src/codebase_qa/eval/`: question sets, metrics, runner, reports.
- `tests/` mirrors the package. `tests/fixtures/` holds chunker inputs and is excluded from lint and type checks.
- `eval/questions/` and `eval/configs/`: eval inputs. `eval/.cache/` is local and git-ignored.

## Invariants

- **The SQLite catalog is the system of record.** Chroma collections are derived from it. Write vectors before the catalog transaction commits, never after.
- **Only `vector_index.py` imports `chromadb`.** Only `embeddings.py` and `retrieval/rerank.py` import `sentence_transformers` or `torch`, and only inside functions or methods.
- **Keep heavy and optional imports lazy:** torch, sentence-transformers, transformers, anthropic, matplotlib. `import codebase_qa` must not load any of them, and must work without the optional extras.
- **Tests never download models.** Use `HashEmbedder` and `whitespace_token_count`. Mark any test that loads a real model with `@pytest.mark.model`.
- **Bump `SCHEMA_VERSION` in `store.py` on any catalog schema change.** Old indexes are then refused with a rebuild hint; there are no migrations before 1.0.
- **Chunk boundaries come from one reference tokenizer and a fixed 256-token budget.** Don't make chunking depend on the embedding model.
- **Filters apply inside each retriever.** Never post-filter a top-k list unless you document the shortfall.
- **pyright runs in strict mode.** Keep `# pyright: ignore[...]` narrow, rule-specific, and confined to adapters around untyped libraries.
- **`tree-sitter` is pinned `<0.26`**, because 0.26.0 segfaults under repeated query matching. `test_repeated_parsing_is_stable` guards this. Re-check it before lifting the pin.
- **Model presets with custom remote code load only at a pinned revision.**

## Conventions

- Match the surrounding code. Comments are rare and explain non-obvious intent. Docstrings are written for maintainers.
- Update `README.md` or `docs/architecture.md` in the same change as any behaviour they describe (commands, flags, defaults, invariants).
- **Don't edit eval question sets to fit results.** Questions and gold targets change only to fix a factual error, and the reason goes in the commit message.
- Never commit `.env`, index directories (`.codebase-qa/`), `eval/.cache/` or downloaded models.
- Commits use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), in the imperative mood, one logical change per commit.
