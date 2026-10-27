"""Chroma-backed vector index.

This is the only module that imports ``chromadb``. Collections are created
without an embedding function: callers always supply vectors, and Chroma
rejects writes that omit them instead of embedding with its default model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import chromadb
import numpy as np
from chromadb import Collection

from codebase_qa.embeddings import Vectors

Space = Literal["cosine", "l2", "ip"]
MetadataValue = str | int | float | bool
Metadata = Mapping[str, MetadataValue]
Where = Mapping[str, Any]


@dataclass(frozen=True)
class HnswConfig:
    """HNSW build and search parameters.

    ``space``, ``ef_construction`` and ``max_neighbors`` are fixed once a
    collection exists; ``ef_search`` can be changed later.
    """

    space: Space = "cosine"
    ef_construction: int = 100
    max_neighbors: int = 16
    ef_search: int = 100


@dataclass(frozen=True)
class VectorHit:
    id: str
    distance: float
    document: str
    metadata: Metadata


class IndexConfigError(ValueError):
    """An existing collection was opened with incompatible HNSW settings."""


class ChromaVectorIndex:
    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    @classmethod
    def open(
        cls,
        path: Path,
        name: str,
        hnsw: HnswConfig | None = None,
    ) -> ChromaVectorIndex:
        """Open or create a persistent collection.

        Reopening with different immutable HNSW settings raises rather than
        silently using the stored ones.
        """
        hnsw = hnsw or HnswConfig()
        client = chromadb.PersistentClient(path=str(path))
        existing = {c.name for c in client.list_collections()}
        if name in existing:
            collection = client.get_collection(name, embedding_function=None)
            stored = _hnsw_settings(collection)
            for key in ("space", "ef_construction", "max_neighbors"):
                if stored.get(key) != getattr(hnsw, key):
                    raise IndexConfigError(
                        f"collection {name!r} has {key}={stored.get(key)!r}, "
                        f"requested {getattr(hnsw, key)!r}"
                    )
            index = cls(collection)
            if stored.get("ef_search") != hnsw.ef_search:
                index.set_ef_search(hnsw.ef_search)
            return index
        collection = client.create_collection(
            name,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": hnsw.space,
                    "ef_construction": hnsw.ef_construction,
                    "max_neighbors": hnsw.max_neighbors,
                    "ef_search": hnsw.ef_search,
                }
            },
        )
        return cls(collection)

    @property
    def name(self) -> str:
        return self._collection.name

    @property
    def ef_search(self) -> int:
        return int(_hnsw_settings(self._collection)["ef_search"])

    def count(self) -> int:
        return self._collection.count()

    def set_ef_search(self, ef_search: int) -> None:
        """Persist a new query-time beam width.

        Chroma reads ``ef_search`` when a process first loads the collection's
        HNSW segment (its first query). A process that has already queried the
        collection keeps the old value until it restarts, so set this before
        querying.
        """
        self._collection.modify(configuration={"hnsw": {"ef_search": ef_search}})

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Vectors,
        documents: Sequence[str],
        metadatas: Sequence[Metadata],
    ) -> None:
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids, embeddings, documents and metadatas differ in length")
        if not ids:
            return
        self._collection.upsert(
            ids=list(ids),
            embeddings=embeddings,
            documents=list(documents),
            metadatas=[dict(m) for m in metadatas],
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._collection.delete(ids=list(ids))

    def get_embeddings(self, ids: Sequence[str]) -> dict[str, Vectors]:
        """Stored vectors for the ids that exist; missing ids are omitted."""
        if not ids:
            return {}
        result = self._collection.get(ids=list(ids), include=["embeddings"])
        embeddings = result["embeddings"]
        if embeddings is None:
            return {}
        return {
            id_: np.asarray(vector, dtype=np.float32)
            for id_, vector in zip(result["ids"], embeddings, strict=True)
        }

    def query(self, embedding: Vectors, k: int, where: Where | None = None) -> list[VectorHit]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=cast(Any, dict(where)) if where else None,
            include=["distances", "documents", "metadatas"],
        )
        ids = result["ids"][0]
        distances = (result["distances"] or [[]])[0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        return [
            VectorHit(
                id=id_,
                distance=float(distance),
                document=document,
                metadata=cast(Metadata, dict(metadata)),
            )
            for id_, distance, document, metadata in zip(
                ids, distances, documents, metadatas, strict=True
            )
        ]


def _hnsw_settings(collection: Collection) -> dict[str, Any]:
    hnsw = collection.configuration.get("hnsw")  # pyright: ignore[reportUnknownMemberType]
    return dict(cast(Mapping[str, Any], hnsw or {}))
