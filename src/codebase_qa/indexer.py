"""Builds the catalog and vector collections for one repository revision."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from codebase_qa.chunking import (
    Chunk,
    ChunkerName,
    FileRoute,
    LanguageRegistry,
    TokenCounter,
    chunk_file,
    route_file,
)
from codebase_qa.embeddings import Embedder
from codebase_qa.git import GitRepo, TreeEntry
from codebase_qa.store import Build, FileRecord, Store, StoredChunk
from codebase_qa.vector_index import ChromaVectorIndex

CATALOG_FILE = "catalog.sqlite3"
VECTORS_DIR = "vectors"
DEFAULT_INDEX_DIR = ".codebase-qa"

ProgressCallback = Callable[[int, int], None]
"""Called with (chunks embedded so far, total chunks to embed)."""


@dataclass(frozen=True)
class IndexOptions:
    chunker: ChunkerName = "treesitter"
    max_tokens: int = 256
    excludes: tuple[str, ...] = ()
    batch_size: int = 64


@dataclass(frozen=True)
class IndexReport:
    commit: str
    collection: str
    files: int
    skipped: int
    chunks: int
    embedded: int
    up_to_date: bool


@dataclass(frozen=True)
class BuildStatus:
    build: Build
    vectors: int
    stale: bool


@dataclass(frozen=True)
class IndexStats:
    current_rev: str | None
    files: int
    chunks: dict[ChunkerName, int]
    languages: dict[str, int]
    kinds: dict[str, int]
    builds: list[BuildStatus] = field(default_factory=lambda: list[BuildStatus]())


def collection_name(model_id: str, revision: str | None, chunker: ChunkerName) -> str:
    """Chroma collection for one (model, model revision, chunker) build.

    The revision is part of the name so that re-pinning a model creates a new
    collection instead of mixing vectors from two weight versions.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.rsplit("/", 1)[-1].lower()).strip("-")
    return f"{slug or 'model'}-{(revision or 'unpinned')[:8]}-{chunker}"


def index_repo(
    repo: Path,
    index_dir: Path,
    embedder: Embedder,
    token_count: TokenCounter,
    options: IndexOptions | None = None,
    *,
    rev: str = "HEAD",
    on_progress: ProgressCallback | None = None,
) -> IndexReport:
    """Index ``rev`` of ``repo`` into ``index_dir`` for one embedder and chunker.

    Vectors are written to the collection before the catalog transaction
    commits. A crash in between leaves orphan vectors with deterministic ids,
    which the next run overwrites; the catalog never references missing
    vectors.
    """
    options = options or IndexOptions()
    git = GitRepo(repo)
    commit = git.rev_parse(rev)
    store = Store.open(index_dir / CATALOG_FILE)
    vectors_dir = index_dir / VECTORS_DIR
    name = collection_name(embedder.model_id, embedder.revision, options.chunker)
    try:
        existing = {b.collection: b for b in store.builds()}
        build = existing.get(name)
        if store.current_rev == commit and build is not None and build.indexed_rev == commit:
            files = len(store.files())
            chunks = store.chunk_count(options.chunker)
            return IndexReport(commit, name, files, 0, chunks, 0, up_to_date=True)

        entries = git.ls_tree(commit)
        registry = LanguageRegistry.default()
        routed = {
            e.path: (e, r)
            for e in entries
            if (r := route_file(e.path, e.size, registry=registry, excludes=options.excludes))
        }

        with store.transaction():
            if store.current_rev != commit:
                store.delete_chunks("treesitter")
                store.delete_chunks("window")
                store.replace_files(
                    FileRecord(e.path, e.blob_sha, r.language, e.size) for e, r in routed.values()
                )
                store.set_current_rev(commit)
            if store.chunk_count(options.chunker) == 0:
                store.insert_chunks(_chunk_files(git, routed, options, token_count, registry))
            chunks = store.chunks(options.chunker)
            embedded = _embed_into(
                vectors_dir, name, chunks, commit, embedder, options.batch_size, on_progress
            )
            store.record_build(
                Build(
                    collection=name,
                    model_id=embedder.model_id,
                    model_revision=embedder.revision or "",
                    chunker=options.chunker,
                    dimension=embedder.dimension,
                    indexed_rev=commit,
                    indexed_at="",
                )
            )
        return IndexReport(
            commit,
            name,
            files=len(routed),
            skipped=len(entries) - len(routed),
            chunks=len(chunks),
            embedded=embedded,
            up_to_date=False,
        )
    finally:
        store.close()


def index_stats(index_dir: Path) -> IndexStats:
    store = Store.open(index_dir / CATALOG_FILE)
    try:
        current = store.current_rev
        collections = set(ChromaVectorIndex.collection_names(index_dir / VECTORS_DIR))
        statuses = [
            BuildStatus(
                build=b,
                vectors=(
                    ChromaVectorIndex.open(index_dir / VECTORS_DIR, b.collection).count()
                    if b.collection in collections
                    else 0
                ),
                stale=b.indexed_rev != current,
            )
            for b in store.builds()
        ]
        return IndexStats(
            current_rev=current,
            files=len(store.files()),
            chunks={c: store.chunk_count(c) for c in ("treesitter", "window")},
            languages=store.counts_by("language", "treesitter"),
            kinds=store.counts_by("symbol_kind", "treesitter"),
            builds=statuses,
        )
    finally:
        store.close()


def _chunk_files(
    git: GitRepo,
    routed: dict[str, tuple[TreeEntry, FileRoute]],
    options: IndexOptions,
    token_count: TokenCounter,
    registry: LanguageRegistry,
) -> list[Chunk]:
    paths_by_blob: dict[str, list[str]] = defaultdict(list)
    for path, (entry, _) in routed.items():
        paths_by_blob[entry.blob_sha].append(path)
    chunks: list[Chunk] = []
    for sha, data in git.read_blobs(paths_by_blob):
        source = data.decode("utf-8", errors="replace")
        for path in paths_by_blob[sha]:
            _, route = routed[path]
            chunks += chunk_file(
                path,
                source,
                route,
                chunker=options.chunker,
                token_count=token_count,
                max_tokens=options.max_tokens,
                registry=registry,
            )
    return chunks


def _embed_into(
    vectors_dir: Path,
    name: str,
    chunks: list[Chunk],
    commit: str,
    embedder: Embedder,
    batch_size: int,
    on_progress: ProgressCallback | None,
) -> int:
    ChromaVectorIndex.drop(vectors_dir, name)
    index = ChromaVectorIndex.open(vectors_dir, name)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([c.embedding_text for c in batch])
        index.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],
            metadatas=[StoredChunk(c, commit).metadata for c in batch],
        )
        if on_progress is not None:
            on_progress(start + len(batch), len(chunks))
    return len(chunks)
