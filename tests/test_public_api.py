"""The README's library example, with the model swapped for the test embedder."""

from conftest import GitRepoFactory

import codebase_qa
from codebase_qa import Embedder, RetrievalConfig, Retriever, SearchResult, index_repo
from codebase_qa.chunking import whitespace_token_count
from codebase_qa.embeddings import HashEmbedder


def test_readme_library_example(make_repo: GitRepoFactory) -> None:
    repo = make_repo(
        {
            "client.py": (
                "class Client:\n"
                "    def set_timeout(self, seconds):\n"
                '        """Configure the request timeout."""\n'
                "        self.timeout = seconds\n"
            ),
            "README.md": "# Example\n\nA client with configurable timeouts.\n",
        }
    )
    index_dir = repo / ".codebase-qa"
    embedder: Embedder = HashEmbedder()
    index_repo(repo, index_dir, embedder, whitespace_token_count)

    with Retriever.open(index_dir, embedder=lambda: embedder) as retriever:
        config = RetrievalConfig(mode="hybrid", k=5)
        results = retriever.retrieve("how are request timeouts configured", config)

    assert results
    assert all(isinstance(r, SearchResult) for r in results)
    assert "Client.set_timeout" in {r.chunk.symbol_path for r in results}
    assert results[0].location.count(":") == 1


def test_public_names_are_exported() -> None:
    assert set(codebase_qa.__all__) == {
        "Embedder",
        "RetrievalConfig",
        "Retriever",
        "SearchResult",
        "index_repo",
    }
