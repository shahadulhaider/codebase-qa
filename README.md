# codebase-qa

Index a Git repository and answer questions about it with cited sources.

`codebase-qa` splits source files into syntax-aware chunks (classes, functions, methods), embeds them with a local model, and indexes them for both dense and keyword search. You can search the index directly, or ask a question and get a streamed answer whose `[n]` citations point at exact `file:start-end` ranges of the indexed commit.

- **Syntax-aware chunking** with tree-sitter for Python, TypeScript/TSX and JavaScript/JSX. Line windows for Markdown, reStructuredText, JSON, YAML, TOML, HTML and CSS.
- **Hybrid retrieval**: dense vectors (Chroma), BM25 (SQLite FTS5), reciprocal rank fusion, and optional cross-encoder reranking.
- **Filters** by language, symbol kind, file role (source/test/docs/config) and path glob, applied inside every retriever.
- **Incremental indexing**: unchanged files are never re-read, and unchanged chunks are never re-embedded.
- **Validated citations**: references to sources the model was not given are removed as the answer streams.
- **Local by default**: embeddings run on your machine, and the default answer model is a local Ollama server.
- **Retrieval evaluation** harness with hand-written question sets.

How it works, and why it is built this way: [docs/architecture.md](docs/architecture.md).

## Install

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and `git`.

```sh
git clone <this repository> codebase-qa
cd codebase-qa
uv sync
```

On Linux, PyTorch is installed from the CPU-only wheel index (about 1 GB instead of several GB of CUDA wheels). On macOS, the standard wheels are used, and Apple Silicon GPUs are picked up automatically (`mps`).

Optional extras:

| Extra | Adds | Install |
|---|---|---|
| `anthropic` | The Anthropic answer backend | `uv sync --extra anthropic` |
| `eval` | The evaluation harness (matplotlib, PyYAML) | `uv sync --extra eval` |

The embedding model (default `sentence-transformers/all-MiniLM-L6-v2`, about 90 MB) is downloaded from Hugging Face on first use. Pass `--offline` to use only cached models.

To run `codebase-qa` from any directory, point `uv run` at the clone:

```sh
alias codebase-qa="uv run --project /path/to/codebase-qa codebase-qa"
```

## Quickstart

From the repository you want to query:

```sh
codebase-qa index                     # index HEAD into ./.codebase-qa/
codebase-qa stats                     # files, chunks, collections
codebase-qa search "how are request timeouts configured" --mode hybrid
codebase-qa ask "how are request timeouts configured" --mode hybrid
```

`ask` needs an answer model. The default is a local [Ollama](https://ollama.com) server:

```sh
ollama pull qwen2.5-coder:7b
```

If no model server is reachable, `ask` exits with: "Start the model server or point --base-url at one. `codebase-qa search` works without a model."

## Usage

Every command prints its full options with `--help`. `REPO` defaults to the current directory, and the index lives in `REPO/.codebase-qa/` unless `--index-dir` says otherwise.

### `index`: build or update an index

```sh
codebase-qa index [REPO] [--rev HEAD] [--chunker treesitter|window] [--model minilm]
```

- `--rev` indexes any commit, branch or tag. Only committed content is indexed; uncommitted changes are ignored.
- Re-running `index` after new commits processes only changed files. It reports `+added ~modified -deleted =unchanged` files and how many chunks were embedded or reused.
- `--model` takes a built-in preset (`minilm`, the default, or `coderankembed`) or a Hugging Face model id. Models that ship custom code need `--trust-remote-code`, and should be pinned with `--revision`.
- `--exclude GLOB` skips more paths (repeatable). Lockfiles, minified files, files over 1 MB, and `dist/`, `build/`, `vendor/` and `node_modules/` are always skipped.
- `--device cpu|cuda|mps` overrides automatic device selection.
- `--verify` reconciles the vector store with the catalog even when the index looks current.
- `--rebuild` deletes the index and starts over. It is required after a schema change.

Each (model, chunker) pair is a separate *build* in the same index directory. If you index new commits with one build, the others become *stale*: `stats` shows them, and `search` asks you to re-index them.

### `search`: ranked chunks, no language model needed

```sh
codebase-qa search "QUERY" [REPO] [-k 10] [--mode dense|bm25|hybrid] [--rerank]
```

- `--mode dense` (the default) finds paraphrases. `bm25` finds exact identifiers. `hybrid` fuses both.
- `--rerank` reorders the top `--rerank-k` results (default 20) with a cross-encoder. It is slower, and often more precise for natural-language questions.
- Filters, all applied before ranking:
  - `--lang python` (repeatable);
  - `--kind function|method|class|module|...` (repeatable);
  - `--role source|test|docs|config` (repeatable);
  - `--path 'src/*.py'` (SQLite GLOB syntax: case-sensitive, and `*` crosses `/`).
- `--chunker` and `--model` pick a build when an index holds several. Otherwise the most recently indexed build is used.
- `--show` prints each result's text.

The table shows each result's location, symbol and kind, plus per-retriever ranks and scores, so you can see why it ranked where it did.

### `ask`: answer with citations

```sh
codebase-qa ask "QUESTION" [REPO] [-k 8] [--mode dense|bm25|hybrid] [--backend openai-compatible|anthropic]
```

`ask` accepts the same retrieval options as `search`, then sends the top `k` chunks to an answer model. The answer streams to stdout, followed by the cited sources. Warnings about citations that pointed at unknown sources go to stderr. `--show-context` prints the exact prompt sent to the model.

Answer backends:

| Backend | Setup | Code leaves your machine? |
|---|---|---|
| Ollama (default) | `ollama pull qwen2.5-coder:7b` | No |
| llama.cpp server | `llama-server -hf <repo> --reasoning off`, then `--base-url http://localhost:8080/v1` | No |
| Hosted OpenAI-compatible API (OpenRouter, DeepInfra, …) | `--base-url`, `--llm-model`, `CODEBASE_QA_API_KEY` | **Yes**: retrieved chunks are sent to the provider |
| Anthropic | `uv sync --extra anthropic`, `--backend anthropic`, `ANTHROPIC_API_KEY` | **Yes** |

[`.env.example`](.env.example) has ready-made settings for each backend. Copy it to `.env` (git-ignored) and run with `uv run --env-file .env codebase-qa ask ...`.

### `stats` and `chunk`

- `codebase-qa stats [REPO]` shows the indexed revision, chunk counts by language and kind, and each build's vector count and fresh/stale state.
- `codebase-qa chunk FILE [--chunker window] [--show]` shows how one file would be split, with each chunk's token count against the 256-token budget. It needs no index.

### Configuration

Options resolve in this order: command-line flag, then environment variable, then the settings the index was built with, then the built-in default. There is no config file.

| Variable | Option |
|---|---|
| `CODEBASE_QA_INDEX_DIR` | `--index-dir` |
| `CODEBASE_QA_MODEL` | `--model` (embedding model) |
| `CODEBASE_QA_DEVICE` | `--device` |
| `CODEBASE_QA_BACKEND` | `--backend` |
| `CODEBASE_QA_BASE_URL` | `--base-url` |
| `CODEBASE_QA_LLM_MODEL` | `--llm-model` (answer model) |
| `CODEBASE_QA_API_KEY` | API key for hosted OpenAI-compatible endpoints (env only) |
| `ANTHROPIC_API_KEY` | API key for the Anthropic backend (env only) |
| `CODEBASE_QA_EVAL_CACHE` | `eval --cache-dir` |

### Library API

The CLI is a thin layer over a small Python API:

```python
from pathlib import Path

from codebase_qa import RetrievalConfig, Retriever, index_repo
from codebase_qa.embeddings import PRESETS, SentenceTransformerEmbedder, load_token_counter

repo = Path(".")
index_dir = repo / ".codebase-qa"
embedder = SentenceTransformerEmbedder.from_preset("minilm")
index_repo(repo, index_dir, embedder, load_token_counter(PRESETS["minilm"]))

with Retriever.open(index_dir, embedder=lambda: embedder) as retriever:
    config = RetrievalConfig(mode="hybrid", k=5)
    for result in retriever.retrieve("how are request timeouts configured", config):
        print(result.location, result.chunk.symbol_path, f"{result.score:.3f}")
```

`index_repo`, `Retriever`, `RetrievalConfig`, `SearchResult` and the `Embedder` protocol are exported from `codebase_qa` and kept stable. Any object implementing `Embedder` can replace the sentence-transformers model.

## Evaluation

`codebase-qa eval` measures retrieval quality: does the right code reach the top k? Answers are not graded, because an answer can only cite what retrieval returned.

- **Questions:**
  - [`eval/questions/`](eval/questions) holds about 30 hand-written questions each for [httpx](https://github.com/encode/httpx) (Python) and [zod](https://github.com/colinhacks/zod) (TypeScript), pinned to a release commit.
  - Each question has gold targets: files, and the functions or classes that answer it.
  - Questions are typed as *identifier* (names a symbol), *conceptual* (a paraphrase) or *cross-file* (needs several targets).
  - They were written from the source and docs before any retrieval run, so they don't echo how the index happens to phrase things.
- **Metrics:**
  - recall@1/5/10 and MRR@10, at file level and at symbol level.
  - Several retrieved pieces of one function count once.
  - A line-window chunk counts for every gold symbol it overlaps, so the window baseline is not penalised for lacking symbol names.
- **Stages:**
  - [`eval/configs/stage1.yaml`](eval/configs/stage1.yaml) compares chunkers × embedding models with dense retrieval.
  - [`eval/configs/stage2.yaml`](eval/configs/stage2.yaml) compares dense, BM25 and hybrid retrieval, with and without reranking.

```sh
uv sync --extra eval
uv run codebase-qa eval check eval/questions/*.yaml   # validate gold targets at the pinned commits
uv run codebase-qa eval run eval/configs/stage1.yaml  # writes eval/results/ and docs/eval/
```

Corpora are cloned and indexed once into `eval/.cache/`; later runs reuse those indexes. The stage 1 Qwen3 model is served by Ollama (`ollama pull qwen3-embedding:0.6b`). The full methodology is in [docs/architecture.md](docs/architecture.md#evaluation).

## Development

```sh
uv sync --all-extras
uv run pytest                           # fast suite; never downloads models
uv run pytest -m model                  # also runs real embedding and reranking models
uv run ruff check . && uv run ruff format --check .
uv run pyright                          # strict mode
```

The fast suite uses a deterministic hashing embedder and throwaway Git repositories, so it runs offline in seconds. Tests marked `model` download and run the real models.

### Layout

```text
src/codebase_qa/
  cli.py            command-line interface
  git.py            reads commits through the git CLI
  chunking/         file routing, language registry, tree-sitter and window chunkers
  embeddings.py     Embedder protocol and backends
  store.py          SQLite catalog: files, chunks, FTS5 index, builds
  vector_index.py   Chroma collections (the only module that imports chromadb)
  indexer.py        incremental indexing
  retrieval/        dense, BM25, filters, fusion, reranking
  answer/           prompt, answer backends, citation validation
  eval/             question sets, metrics, runner, reports
tests/              mirrors the package layout
eval/               question sets and stage configs
docs/               architecture and eval charts
```

### Extending

- **A language**: add a `LanguageSpec` to `builtin_specs` in `src/codebase_qa/chunking/languages.py`, with its tree-sitter grammar package and a query file in `chunking/queries/`. The query captures each definition as `@definition.<kind>` and its name as `@name`. The chunker itself needs no changes.
- **An embedding model**:
  - Any `sentence-transformers` model works by Hugging Face id.
  - For another runtime, implement the `Embedder` protocol in `embeddings.py`, returning unit-length float32 vectors.
  - New models and revisions get their own vector collection automatically.
- **An answer backend**: implement `AnswerBackend` (`name`, `model`, `stream`) in `answer/`, and raise `BackendUnavailableError` when the endpoint can't be reached.

Contributor conventions (invariants, commit style) are in [AGENTS.md](AGENTS.md).

## Limitations

- Indexes committed revisions only, one current revision per index directory.
- Code-tuned embedding models are slow on CPU, which is why the default is the small general-purpose MiniLM.
- Citations are checked to point at retrieved sources, not to support the sentence they follow.
- Symbol-aware chunking covers Python, TypeScript and JavaScript; other languages are indexed as line windows, or skipped if not listed above.

## License

[MIT](LICENSE)
