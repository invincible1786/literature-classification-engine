"""Live smoke test against Europe PMC REST API (marked @pytest.mark.live).
Excluded from default local CI runs. Run with: pytest -m live.
"""
import pytest
from app.retrieval import search_europe_pmc, build_query
from app.classifier import classify_publications
from app.nli import DebertaNLIScorer


@pytest.mark.live
def test_live_vanillin_europe_pmc():
    """Live integration test querying Europe PMC directly for 'vanillin'."""
    scorer = DebertaNLIScorer()
    query = build_query("vanillin")
    pubs = search_europe_pmc(query, max_results=5)
    assert len(pubs) > 0

    res = classify_publications(
        publications=pubs,
        entity="vanillin",
        criterion="The paper describes the synthesis or production of vanillin.",
        scorer=scorer
    )
    assert len(res.results) == len(pubs)
    for r in res.results:
        assert r["classification"] in ("qualifies", "does_not_qualify")
        if r["evidence_sentence"]:
            assert r["evidence_status"] in ("supported", "found", "contradicted")

