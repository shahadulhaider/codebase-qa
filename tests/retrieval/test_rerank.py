import pytest

from codebase_qa.retrieval import CrossEncoderReranker

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(device="cpu")


def test_relevant_code_outscores_unrelated_code(reranker: CrossEncoderReranker) -> None:
    query = "where are proxy settings read from environment variables"
    relevant = (
        "def get_environment_proxies():\n"
        "    proxy_info = getproxies()\n"
        "    for scheme in ('http', 'https', 'all'):\n"
        "        if proxy_info.get(scheme):\n"
        "            mounts[f'{scheme}://'] = proxy_info[scheme]\n"
    )
    unrelated = "def format_date(value):\n    return value.strftime('%Y-%m-%d')\n"

    scores = reranker.score(query, [unrelated, relevant])

    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_scores_do_not_depend_on_batch_composition(reranker: CrossEncoderReranker) -> None:
    texts = ["def close(self): self.pool.close()", "class Client: pass", "x = 1"]

    together = reranker.score("close the connection pool", texts)
    alone = [reranker.score("close the connection pool", [t])[0] for t in texts]

    assert together == pytest.approx(alone, abs=1e-4)


def test_long_inputs_are_truncated_not_rejected(reranker: CrossEncoderReranker) -> None:
    long_text = "def handler(request):\n" + "    value = compute(value)\n" * 400

    assert len(reranker.score("request handler", [long_text])) == 1


def test_empty_input_needs_no_model_call(reranker: CrossEncoderReranker) -> None:
    assert reranker.score("anything", []) == []
