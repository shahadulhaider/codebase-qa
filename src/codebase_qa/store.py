"""SQLite catalog: the system of record for files, chunks, builds and BM25.

Vector collections are derived from this catalog and can be rebuilt from
it. The catalog tracks a single current revision; builds indexed at any
other revision are stale.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from codebase_qa.chunking import Chunk, ChunkerName, SymbolKind
from codebase_qa.terms import search_terms

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files (
    path TEXT PRIMARY KEY,
    blob_sha TEXT NOT NULL,
    language TEXT NOT NULL,
    size INTEGER NOT NULL
);
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    chunker TEXT NOT NULL,
    file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    language TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_path TEXT NOT NULL,
    symbol_kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content_sha TEXT NOT NULL,
    text TEXT NOT NULL,
    UNIQUE (chunker, chunk_id)
);
CREATE INDEX chunks_by_file ON chunks (chunker, file_path);
CREATE INDEX chunks_by_content ON chunks (content_sha);
CREATE VIRTUAL TABLE chunks_fts USING fts5(terms, path, tokenize = 'unicode61');
CREATE TABLE builds (
    collection TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    chunker TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    indexed_rev TEXT NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


class SchemaVersionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileRecord:
    path: str
    blob_sha: str
    language: str
    size: int


@dataclass(frozen=True)
class Build:
    collection: str
    model_id: str
    model_revision: str
    chunker: ChunkerName
    dimension: int
    indexed_rev: str
    indexed_at: str


@dataclass(frozen=True)
class StoredChunk:
    chunk: Chunk
    commit: str

    @property
    def metadata(self) -> dict[str, str | int]:
        """Vector-store metadata; every value is a scalar, never None."""
        c = self.chunk
        return {
            "file_path": c.file_path,
            "language": c.language,
            "symbol_name": c.symbol_name,
            "symbol_path": c.symbol_path,
            "symbol_kind": c.symbol_kind,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "chunker": c.chunker,
            "commit": self.commit,
            "content_sha": c.content_sha,
        }


class Store:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    @classmethod
    def open(cls, path: Path) -> Store:
        """Open or create the catalog, refusing an incompatible schema version."""
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, isolation_level=None)
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA foreign_keys = ON")
        version = int(db.execute("PRAGMA user_version").fetchone()[0])
        if version == 0 and not _has_tables(db):
            db.executescript(f"BEGIN; {_SCHEMA} PRAGMA user_version = {SCHEMA_VERSION}; COMMIT;")
        elif version != SCHEMA_VERSION:
            db.close()
            raise SchemaVersionError(
                f"index at {path.parent} uses schema {version}, this version needs "
                f"{SCHEMA_VERSION}; rebuild it with `codebase-qa index --rebuild`"
            )
        return cls(db)

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def transaction(self) -> Generator[None]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    @property
    def current_rev(self) -> str | None:
        return self.get_meta("current_rev")

    def set_current_rev(self, rev: str) -> None:
        self.set_meta("current_rev", rev)

    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def chunkers(self) -> set[ChunkerName]:
        """Chunkers that have chunks in the catalog."""
        return {_chunker(r[0]) for r in self._db.execute("SELECT DISTINCT chunker FROM chunks")}

    def files(self) -> dict[str, FileRecord]:
        rows = self._db.execute("SELECT path, blob_sha, language, size FROM files")
        return {row[0]: FileRecord(*row) for row in rows}

    def replace_files(self, records: Iterable[FileRecord]) -> None:
        """Make ``records`` the current file set; chunks of removed files cascade."""
        records = list(records)
        keep = {r.path for r in records}
        for path in set(self.files()) - keep:
            self.delete_file(path)
        self._db.executemany(
            "INSERT INTO files (path, blob_sha, language, size) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (path) DO UPDATE SET blob_sha = excluded.blob_sha, "
            "language = excluded.language, size = excluded.size",
            [(r.path, r.blob_sha, r.language, r.size) for r in records],
        )

    def delete_file(self, path: str) -> None:
        self._delete_fts("SELECT id FROM chunks WHERE file_path = ?", (path,))
        self._db.execute("DELETE FROM files WHERE path = ?", (path,))

    def delete_chunks(self, chunker: ChunkerName, file_path: str | None = None) -> None:
        where, params = "chunker = ?", [chunker]
        if file_path is not None:
            where, params = f"{where} AND file_path = ?", [chunker, file_path]
        self._delete_fts(f"SELECT id FROM chunks WHERE {where}", params)
        self._db.execute(f"DELETE FROM chunks WHERE {where}", params)

    def insert_chunks(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            cursor = self._db.execute(
                "INSERT INTO chunks (chunk_id, chunker, file_path, language, symbol_name, "
                "symbol_path, symbol_kind, start_line, end_line, content_sha, text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    c.chunk_id,
                    c.chunker,
                    c.file_path,
                    c.language,
                    c.symbol_name,
                    c.symbol_path,
                    c.symbol_kind,
                    c.start_line,
                    c.end_line,
                    c.content_sha,
                    c.text,
                ),
            )
            self._db.execute(
                "INSERT INTO chunks_fts (rowid, terms, path) VALUES (?, ?, ?)",
                (cursor.lastrowid, search_terms(c.text), c.file_path),
            )

    def chunks(self, chunker: ChunkerName) -> list[Chunk]:
        rows = self._db.execute(
            "SELECT file_path, language, text, start_line, end_line, symbol_kind, chunker, "
            "symbol_name, symbol_path FROM chunks WHERE chunker = ? ORDER BY file_path, start_line",
            (chunker,),
        )
        return [
            Chunk(
                file_path=r[0],
                language=r[1],
                text=r[2],
                start_line=r[3],
                end_line=r[4],
                symbol_kind=_kind(r[5]),
                chunker=_chunker(r[6]),
                symbol_name=r[7],
                symbol_path=r[8],
            )
            for r in rows
        ]

    def chunk_count(self, chunker: ChunkerName) -> int:
        row = self._db.execute("SELECT count(*) FROM chunks WHERE chunker = ?", (chunker,))
        return int(row.fetchone()[0])

    def counts_by(self, column: str, chunker: ChunkerName) -> dict[str, int]:
        if column not in {"language", "symbol_kind"}:
            raise ValueError(column)
        rows = self._db.execute(
            f"SELECT {column}, count(*) FROM chunks WHERE chunker = ? GROUP BY 1 ORDER BY 2 DESC",
            (chunker,),
        )
        return {str(k): int(n) for k, n in rows}

    def builds(self) -> list[Build]:
        rows = self._db.execute(
            "SELECT collection, model_id, model_revision, chunker, dimension, indexed_rev, "
            "indexed_at FROM builds ORDER BY indexed_at DESC"
        )
        return [Build(r[0], r[1], r[2], _chunker(r[3]), int(r[4]), r[5], r[6]) for r in rows]

    def record_build(self, build: Build) -> None:
        self._db.execute(
            "INSERT INTO builds (collection, model_id, model_revision, chunker, dimension, "
            "indexed_rev) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (collection) DO UPDATE SET "
            "indexed_rev = excluded.indexed_rev, indexed_at = excluded.indexed_at",
            (
                build.collection,
                build.model_id,
                build.model_revision,
                build.chunker,
                build.dimension,
                build.indexed_rev,
            ),
        )

    def _delete_fts(self, select_ids: str, params: Sequence[object]) -> None:
        self._db.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({select_ids})", params)


def _has_tables(db: sqlite3.Connection) -> bool:
    return db.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0] > 0


def _kind(value: str) -> SymbolKind:
    return cast(SymbolKind, value)


def _chunker(value: str) -> ChunkerName:
    return cast(ChunkerName, value)
