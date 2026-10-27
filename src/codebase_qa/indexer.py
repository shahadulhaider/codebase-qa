"""Builds and incrementally updates the catalog and vector collections.

Indexing has two steps. First the catalog is brought to the target
revision: files are diffed by blob SHA and only added or modified files are
re-chunked, for every chunker present in the catalog. Then the build's
collection is synced to the catalog: ids the catalog no longer has are
deleted, missing ids reuse an existing vector with the same ``content_sha``
when there is one, and only genuinely new text is embedded. The same sync
repairs stale builds and drift between the two stores.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from codebase_qa.chunking import (
    Chunk,
    ChunkerName,
    FileRoute,
    LanguageRegistry,
    TokenCounter,
    chunk_file,
    route_file,
)
from codebase_qa.embeddings import Embedder, Vectors
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
    verify: bool = False
    """Sync the collection even when the build is recorded as up to date."""

    @property
    def chunk_settings(self) -> str:
        """Settings that change chunk boundaries; a change forces re-chunking."""
        return json.dumps({"max_tokens": self.max_tokens, "excludes": sorted(self.excludes)})


@dataclass(frozen=True)
class FileChanges:
    added: int = 0
    modified: int = 0
    deleted: int = 0
    unchanged: int = 0


@dataclass(frozen=True)
class SyncReport:
    embedded: int = 0
    reused: int = 0
    removed: int = 0
    kept: int = 0


@dataclass(frozen=True)
class IndexReport:
    commit: str
    collection: str
    files: int
    skipped: int
    chunks: int
    changes: FileChanges = field(default_factory=FileChanges)
    sync: SyncReport = field(default_factory=SyncReport)
    up_to_date: bool = False


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


def is_up_to_date(
    repo: Path,
    index_dir: Path,
    model_id: str,
    revision: str | None,
    options: IndexOptions,
    *,
    rev: str = "HEAD",
) -> bool:
    """Whether indexing would be a no-op; cheap enough to call before loading a model."""
    if options.verify or not (index_dir / CATALOG_FILE).exists():
        return False
    commit = GitRepo(repo).rev_parse(rev)
    store = Store.open(index_dir / CATALOG_FILE)
    try:
        name = collection_name(model_id, revision, options.chunker)
        return _is_current(store, commit, name, options)
    finally:
        store.close()


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
    """Bring ``index_dir`` to ``rev`` of ``repo`` for one embedder and chunker.

    Vectors are written before the catalog transaction commits. A crash in
    between leaves at most vectors the catalog does not reference, which the
    next sync deletes or overwrites; the catalog never references missing
    vectors.
    """
    options = options or IndexOptions()
    git = GitRepo(repo)
    commit = git.rev_parse(rev)
    store = Store.open(index_dir / CATALOG_FILE)
    name = collection_name(embedder.model_id, embedder.revision, options.chunker)
    try:
        if not options.verify and _is_current(store, commit, name, options):
            files = len(store.files())
            chunks = store.chunk_count(options.chunker)
            return IndexReport(commit, name, files, 0, chunks, up_to_date=True)

        entries = git.ls_tree(commit)
        registry = LanguageRegistry.default()
        routed = {
            e.path: (e, r)
            for e in entries
            if (r := route_file(e.path, e.size, registry=registry, excludes=options.excludes))
        }
        with store.transaction():
            changes = _update_catalog(store, git, routed, commit, options, token_count, registry)
            chunks = store.chunks(options.chunker)
            sync = _sync_collection(
                ChromaVectorIndex.open(index_dir / VECTORS_DIR, name),
                chunks,
                commit,
                embedder,
                options.batch_size,
                on_progress,
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
            changes=changes,
            sync=sync,
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


def _is_current(store: Store, commit: str, name: str, options: IndexOptions) -> bool:
    build = next((b for b in store.builds() if b.collection == name), None)
    return (
        store.current_rev == commit
        and build is not None
        and build.indexed_rev == commit
        and store.get_meta("chunk_settings") == options.chunk_settings
    )


def _update_catalog(
    store: Store,
    git: GitRepo,
    routed: dict[str, tuple[TreeEntry, FileRoute]],
    commit: str,
    options: IndexOptions,
    token_count: TokenCounter,
    registry: LanguageRegistry,
) -> FileChanges:
    old = store.files()
    settings_changed = store.get_meta("chunk_settings") != options.chunk_settings
    added = routed.keys() - old.keys()
    deleted = old.keys() - routed.keys()
    modified = {p for p in routed.keys() & old.keys() if routed[p][0].blob_sha != old[p].blob_sha}
    store.replace_files(
        FileRecord(e.path, e.blob_sha, r.language, e.size) for e, r in routed.values()
    )

    chunkers = store.chunkers()
    chunkers.add(options.chunker)
    for chunker in sorted(chunkers):
        if settings_changed or store.chunk_count(chunker) == 0:
            store.delete_chunks(chunker)
            paths = set(routed)
        else:
            paths = added | modified
            for path in modified:
                store.delete_chunks(chunker, path)
        selected = {p: routed[p] for p in paths}
        store.insert_chunks(
            _chunk_files(git, selected, chunker, options.max_tokens, token_count, registry)
        )
    store.set_current_rev(commit)
    store.set_meta("chunk_settings", options.chunk_settings)
    return FileChanges(
        added=len(added),
        modified=len(modified),
        deleted=len(deleted),
        unchanged=len(routed) - len(added) - len(modified),
    )


def _chunk_files(
    git: GitRepo,
    routed: dict[str, tuple[TreeEntry, FileRoute]],
    chunker: ChunkerName,
    max_tokens: int,
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
            chunks += chunk_file(
                path,
                source,
                routed[path][1],
                chunker=chunker,
                token_count=token_count,
                max_tokens=max_tokens,
                registry=registry,
            )
    return chunks


def _sync_collection(
    index: ChromaVectorIndex,
    chunks: list[Chunk],
    commit: str,
    embedder: Embedder,
    batch_size: int,
    on_progress: ProgressCallback | None,
) -> SyncReport:
    existing = index.content_shas()
    desired = {c.chunk_id: c for c in chunks}
    missing = [c for chunk_id, c in desired.items() if chunk_id not in existing]
    kept = [c for chunk_id, c in desired.items() if chunk_id in existing]
    obsolete = [chunk_id for chunk_id in existing if chunk_id not in desired]

    source_of: dict[str, str] = {}
    for chunk_id, sha in existing.items():
        source_of.setdefault(sha, chunk_id)
    reusable = [c for c in missing if c.content_sha in source_of]
    to_embed = [c for c in missing if c.content_sha not in source_of]

    # Copy reused vectors before deleting obsolete ids, which may be their source.
    for batch in _batches(reusable, batch_size):
        stored = index.get_embeddings([source_of[c.content_sha] for c in batch])
        _upsert(index, batch, np.stack([stored[source_of[c.content_sha]] for c in batch]), commit)
    done = 0
    for batch in _batches(to_embed, batch_size):
        _upsert(index, batch, embedder.embed_documents([c.embedding_text for c in batch]), commit)
        done += len(batch)
        if on_progress is not None:
            on_progress(done, len(to_embed))
    for batch in _batches(kept, batch_size):
        index.update_metadata(
            [c.chunk_id for c in batch], [StoredChunk(c, commit).metadata for c in batch]
        )
    index.delete(obsolete)
    return SyncReport(
        embedded=len(to_embed), reused=len(reusable), removed=len(obsolete), kept=len(kept)
    )


def _upsert(
    index: ChromaVectorIndex, chunks: Sequence[Chunk], vectors: Vectors, commit: str
) -> None:
    index.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[StoredChunk(c, commit).metadata for c in chunks],
    )


def _batches(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
