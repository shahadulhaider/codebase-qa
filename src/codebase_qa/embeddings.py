"""Text embedding backends.

Vectors are computed here rather than by the vector store, so the store never
embeds anything implicitly and the model that produced a collection is always
known to the catalog. All backends return L2-normalised float32 vectors, which
makes cosine distance, inner-product distance and squared L2 distance
rank-equivalent.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

Vectors = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Maps text to fixed-size, unit-length vectors.

    Queries and documents are embedded through separate methods because
    asymmetric retrieval models expect different input prefixes for each.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_seq_length(self) -> int:
        """Tokens beyond this limit are silently dropped by the model."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors:
        """Return a 1-D vector for a single query."""
        ...


@dataclass(frozen=True)
class EmbedderPreset:
    name: str
    model_id: str
    revision: str | None
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False


# Presets pin an exact model revision. For models that ship custom modelling
# code, the pin is what makes trusting that code a reviewed decision rather
# than a moving target.
PRESETS: dict[str, EmbedderPreset] = {
    p.name: p
    for p in (
        EmbedderPreset(
            name="coderankembed",
            model_id="nomic-ai/CodeRankEmbed",
            revision="3c4b60807d71f79b43f3c4363786d9493691f8b1",
            query_prefix="Represent this query for searching relevant code: ",
            trust_remote_code=True,
        ),
        EmbedderPreset(
            name="minilm",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        ),
    )
}

DEFAULT_PRESET = "minilm"


def select_device(requested: str | None = None) -> str:
    """Pick the fastest available torch device unless one is requested."""
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEmbedder:
    """Embedder backed by a local sentence-transformers model."""

    def __init__(
        self,
        preset: EmbedderPreset,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if preset.trust_remote_code and preset.revision is None:
            warnings.warn(
                f"{preset.model_id} runs custom model code but no revision is pinned",
                stacklevel=2,
            )
        from sentence_transformers import SentenceTransformer

        self._preset = preset
        self._batch_size = batch_size
        self._model: SentenceTransformer = SentenceTransformer(
            preset.model_id,
            revision=preset.revision,
            device=select_device(device),
            trust_remote_code=preset.trust_remote_code,
        )
        if preset.trust_remote_code:
            _restore_extended_attention_mask(self._model)
        dimension = self._model.get_embedding_dimension()
        max_seq_length = self._model.max_seq_length
        if dimension is None or max_seq_length is None:
            raise ValueError(f"{preset.model_id} does not report its dimension and length limit")
        self._dimension = dimension
        self._max_seq_length = max_seq_length

    @classmethod
    def from_preset(
        cls, name: str, *, device: str | None = None, batch_size: int = 32
    ) -> SentenceTransformerEmbedder:
        return cls(PRESETS[name], device=device, batch_size=batch_size)

    @property
    def model_id(self) -> str:
        return self._preset.model_id

    @property
    def revision(self) -> str | None:
        return self._preset.revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Token counts before truncation, including special tokens."""
        encoded = self._model.tokenizer(list(texts), add_special_tokens=True)
        return [len(ids) for ids in encoded["input_ids"]]

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return self._encode([self._preset.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> Vectors:
        return self._encode([self._preset.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> Vectors:
        vectors = self._model.encode(  # pyright: ignore[reportUnknownMemberType]
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _restore_extended_attention_mask(model: SentenceTransformer) -> None:
    """Re-add ``get_extended_attention_mask`` for remote code written against transformers 4.

    transformers 5 removed the helper, but pinned remote modelling code (e.g.
    CodeRankEmbed's NomicBert) still calls it. The replacement reproduces the
    4.x behaviour for 2-D padding masks: broadcast to (batch, 1, 1, seq) and
    turn padded positions into a large negative additive bias.
    """
    import torch

    for module in model.modules():
        cls = type(module)
        if cls.__name__.endswith("Model") and not hasattr(module, "get_extended_attention_mask"):

            def get_extended_attention_mask(
                self: torch.nn.Module,
                attention_mask: torch.Tensor,
                input_shape: tuple[int, ...],
                dtype: torch.dtype | None = None,
            ) -> torch.Tensor:
                dtype = dtype or next(self.parameters()).dtype
                mask = attention_mask[:, None, None, :].to(dtype)
                return (1.0 - mask) * torch.finfo(dtype).min

            cls.get_extended_attention_mask = get_extended_attention_mask  # pyright: ignore[reportAttributeAccessIssue]


_TOKEN = re.compile(r"[A-Za-z0-9]+")


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests.

    Tokens are hashed into signed buckets, so texts sharing vocabulary land
    close together without downloading a model.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self.documents_embedded = 0

    @property
    def model_id(self) -> str:
        return f"hash-{self._dimension}"

    @property
    def revision(self) -> str | None:
        return None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_seq_length(self) -> int:
        return 1_000_000

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.documents_embedded += len(texts)
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> Vectors:
        return self._embed(text)

    def _embed(self, text: str) -> Vectors:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if value & 1 else -1.0
            vector[(value >> 1) % self._dimension] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return vector / norm
