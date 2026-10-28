"""Index a Git repository and answer questions about it with cited sources."""

from codebase_qa.embeddings import Embedder
from codebase_qa.indexer import index_repo
from codebase_qa.retrieval import RetrievalConfig, Retriever, SearchResult

__version__ = "0.1.0"

__all__ = ["Embedder", "RetrievalConfig", "Retriever", "SearchResult", "index_repo"]
