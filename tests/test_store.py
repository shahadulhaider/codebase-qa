import sqlite3
from pathlib import Path

import pytest

from codebase_qa.chunking import Chunk
from codebase_qa.store import FileRecord, SchemaVersionError, Store
from codebase_qa.terms import search_terms, split_identifier


def make_chunk(path: str, text: str, line: int = 1) -> Chunk:
    return Chunk(path, "python", text, line, line, "function", "treesitter", "f", "f")


def fts(store_path: Path, query: str) -> list[str]:
    db = sqlite3.connect(store_path)
    rows = db.execute(
        "SELECT c.file_path FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts)",
        (query,),
    )
    return [r[0] for r in rows]


@pytest.mark.parametrize(
    ("identifier", "parts"),
    [
        ("getUserById", ["get", "user", "by", "id"]),
        ("HTTPClient", ["http", "client"]),
        ("parse_url_v2", ["parse", "url", "v2"]),
        ("getHTTPResponse_code", ["get", "http", "response", "code"]),
    ],
)
def test_split_identifier(identifier: str, parts: list[str]) -> None:
    assert split_identifier(identifier) == parts


def test_search_terms_appends_identifier_parts() -> None:
    terms = search_terms("def get_user(): return fetchUserById(x)")

    assert terms.endswith("fetch user by id get user")


def test_fts_matches_camel_case_parts_and_whole_identifiers(tmp_path: Path) -> None:
    store = Store.open(tmp_path / "catalog.sqlite3")
    with store.transaction():
        store.replace_files(
            [FileRecord("a.ts", "s1", "typescript", 1), FileRecord("b.py", "s2", "python", 1)]
        )
        store.insert_chunks([make_chunk("a.ts", "function getUserById(id) {}")])
        store.insert_chunks([make_chunk("b.py", "def load_config(path): ...")])

    assert fts(tmp_path / "catalog.sqlite3", "user") == ["a.ts"]
    assert fts(tmp_path / "catalog.sqlite3", "getuserbyid") == ["a.ts"]
    assert fts(tmp_path / "catalog.sqlite3", "config") == ["b.py"]
    assert fts(tmp_path / "catalog.sqlite3", "load_config") == ["b.py"]


def test_deleting_a_file_removes_its_chunks_and_fts_rows(tmp_path: Path) -> None:
    store = Store.open(tmp_path / "catalog.sqlite3")
    with store.transaction():
        store.replace_files([FileRecord("a.py", "s1", "python", 1)])
        store.insert_chunks([make_chunk("a.py", "def alpha(): ...")])
    with store.transaction():
        store.replace_files([])

    assert store.chunk_count("treesitter") == 0
    assert fts(tmp_path / "catalog.sqlite3", "alpha") == []


def test_failed_transaction_rolls_back(tmp_path: Path) -> None:
    store = Store.open(tmp_path / "catalog.sqlite3")
    with pytest.raises(RuntimeError), store.transaction():
        store.replace_files([FileRecord("a.py", "s1", "python", 1)])
        raise RuntimeError("boom")

    assert store.files() == {}


def test_schema_version_mismatch_asks_for_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    Store.open(path).close()
    sqlite3.connect(path).execute("PRAGMA user_version = 99")

    with pytest.raises(SchemaVersionError, match="--rebuild"):
        Store.open(path)
