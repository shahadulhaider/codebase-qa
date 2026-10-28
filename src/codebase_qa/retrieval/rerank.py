"""Cross-encoder reranking of a short candidate list."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from codebase_qa.embeddings import select_device

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text, higher is more relevant."""
        ...


class CrossEncoderReranker:
    """Scores (query, text) pairs jointly with a sentence-transformers cross-encoder.

    Input pairs longer than ``max_length`` tokens are truncated, which bounds
    latency: cost grows with sequence length for every candidate.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER,
        *,
        revision: str | None = DEFAULT_RERANKER_REVISION,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model_id
        self._batch_size = batch_size
        self._model: CrossEncoder = CrossEncoder(
            model_id, revision=revision, device=select_device(device), max_length=max_length
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict(  # pyright: ignore[reportUnknownMemberType]
            [(query, text) for text in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]
