"""Query an index: dense, BM25, hybrid fusion, reranking and metadata filters."""

from codebase_qa.retrieval.fusion import rrf
from codebase_qa.retrieval.lexical import fts_query
from codebase_qa.retrieval.rerank import CrossEncoderReranker, Reranker
from codebase_qa.retrieval.retriever import RetrievalConfig, Retriever
from codebase_qa.retrieval.searcher import IndexNotFoundError, Searcher, StaleIndexError
from codebase_qa.retrieval.types import Filters, Mode, SearchResult

__all__ = [
    "CrossEncoderReranker",
    "Filters",
    "IndexNotFoundError",
    "Mode",
    "Reranker",
    "RetrievalConfig",
    "Retriever",
    "SearchResult",
    "Searcher",
    "StaleIndexError",
    "fts_query",
    "rrf",
]
