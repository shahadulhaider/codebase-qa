import pytest

from codebase_qa.answer import CitationFilter, check_citations


def stream(pieces: list[str], sources: int) -> tuple[str, CitationFilter]:
    citations = CitationFilter(sources)
    out = "".join(citations.feed(p) for p in pieces) + citations.flush()
    return out, citations


def test_valid_citations_are_kept_and_recorded_in_first_use_order() -> None:
    text, citations = check_citations("Timeouts are set in the client [2]. See also [1, 2].", 3)

    assert text == "Timeouts are set in the client [2]. See also [1, 2]."
    assert citations.cited == [2, 1]
    assert citations.invalid == []


def test_unknown_source_numbers_are_dropped_with_their_leading_space() -> None:
    text, citations = check_citations("It retries [7]. Proxies come from env [1, 9] here.", 2)

    assert text == "It retries. Proxies come from env [1] here."
    assert citations.cited == [1]
    assert citations.invalid == [7, 9]


def test_zero_is_not_a_source_number() -> None:
    text, citations = check_citations("first [0]", 3)

    assert text == "first"
    assert citations.invalid == [0]


@pytest.mark.parametrize("split", range(1, len("see [12] now")))
def test_a_citation_split_across_chunks_is_resolved_as_a_whole(split: int) -> None:
    whole = "see [12] now"
    text, citations = stream([whole[:split], whole[split:]], sources=3)

    assert text == "see now"
    assert citations.invalid == [12]


def test_code_spans_and_fences_are_never_treated_as_citations() -> None:
    answer = "Use `items[0]` [1].\n```python\nfirst = rows[2]\n```\nDone [2]."
    text, citations = stream(list(answer), sources=1)

    assert text == "Use `items[0]` [1].\n```python\nfirst = rows[2]\n```\nDone."
    assert citations.cited == [1]
    assert citations.invalid == [2]


def test_brackets_that_are_not_citations_pass_through() -> None:
    text, citations = stream(["a [link](x) and [", "b] and [1"], sources=1)

    assert text == "a [link](x) and [b] and [1"
    assert citations.cited == []
