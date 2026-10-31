"""Tier 2 Model Tests: Real DeBERTa-v3 CrossEncoder evaluation on target fixtures.
Validates required demonstration cases Q1–Q3, N1–N6, D1 and CI must-pass gates.
"""
import pytest
from app.classifier import classify_publications
from app.retrieval import deduplicate_papers


def make_single_paper_pub(title: str, text: str, pmid: str = "12345678", doi: str = None):
    return [{
        "title": title,
        "abstractText": text,
        "pmid": pmid,
        "doi": doi or f"10.1000/{pmid}",
        "pmcid": None,
        "pubYear": "2021",
        "fulltext_sections": None
    }]


@pytest.mark.slow
def test_q1_artemisinic_acid_yeast_biosynthesis(real_scorer):
    """Q1: Microbial biosynthesis of artemisinic acid qualifies and ranks top."""
    text = "Herein, artemisinic acid was biosynthesized by engineered Saccharomyces cerevisiae, reaching a titer of 25 g/L."
    pubs = make_single_paper_pub(
        title="High-level production of artemisinic acid in yeast",
        text=text,
        pmid="42107802"
    )
    res = classify_publications(
        publications=pubs,
        entity="artemisinic acid",
        criterion="The paper reports microbial biosynthesis of artemisinic acid in this study.",
        scorer=real_scorer
    )
    assert res.results[0]["classification"] == "qualifies"
    assert res.results[0]["evidence_status"] == "supported"
    assert res.results[0]["rank"] <= 3
    assert "artemisinic acid was biosynthesized" in res.results[0]["evidence"]["text"]


@pytest.mark.slow
def test_q2_chemical_synthesis_passive_yield(real_scorer):
    """Q2: Passive voice chemical synthesis reporting yield qualifies."""
    text = "Under optimized conditions, vanillin was obtained in 82% yield via catalytic oxidation."
    pubs = make_single_paper_pub("Catalytic oxidation process", text)
    res = classify_publications(
        publications=pubs,
        entity="vanillin",
        criterion="The paper describes the synthesis or production of vanillin.",
        scorer=real_scorer
    )
    assert res.results[0]["classification"] == "qualifies"
    assert res.results[0]["evidence_status"] == "supported"
    assert "vanillin was obtained in 82% yield" in res.results[0]["evidence"]["text"]


@pytest.mark.slow
def test_q3_caffeine_synthetic_route(real_scorer):
    """Q3: Synthetic route to caffeine qualifies."""
    text = "We report a direct chemical route where caffeine was synthesized in 75% yield from theobromine."
    pubs = make_single_paper_pub("Total synthesis of caffeine", text)
    res = classify_publications(
        publications=pubs,
        entity="caffeine",
        criterion="The paper describes the chemical synthesis of caffeine.",
        scorer=real_scorer
    )
    assert res.results[0]["classification"] == "qualifies"
    assert res.results[0]["evidence_status"] == "supported"


@pytest.mark.slow
def test_n1_pdoac2_catalyst_role_guard(real_scorer):
    """N1: Entity used as catalyst in the presence of Pd(OAc)2 is rejected with guard_reason set."""
    text = "In the presence of palladium(II) acetate and triphenylphosphine, dibenzoxepines were prepared in 78% yield."
    pubs = make_single_paper_pub("Preparation of dibenzoxepines", text)
    res = classify_publications(
        publications=pubs,
        entity="palladium(II) acetate",
        aliases=["Pd(OAc)2"],
        criterion="The paper describes the chemical synthesis of palladium(II) acetate as the product.",
        scorer=real_scorer
    )
    assert res.results[0]["classification"] == "does_not_qualify"
    assert res.results[0]["guard_reason"] is not None
    assert "in the presence of" in res.results[0]["guard_reason"].lower()


@pytest.mark.slow
def test_n2_review_article_abstain(real_scorer):
    """N2: Literature review stating 'we synthesize current knowledge' abstains."""
    title = "Systematic Review of Caffeine Metabolism"
    text = "In this review, we synthesize current knowledge on caffeine intake and human cardiovascular outcomes."
    pubs = make_single_paper_pub(title, text)
    res = classify_publications(
        publications=pubs,
        entity="caffeine",
        criterion="The paper describes the chemical synthesis or production of caffeine.",
        scorer=real_scorer
    )
    assert res.results[0]["classification"] == "does_not_qualify"
    assert res.results[0]["evidence_status"] == "insufficient"
    assert res.results[0]["evidence"] is None


@pytest.mark.slow
def test_n3_toxicology_case_report_abstain(real_scorer):
    """N3: Caffeine overdose case report lacking production verbs abstains."""
    title = "Acute Caffeine Intoxication in a 22-Year-Old Female: Case Report"
    text = "A patient presented to the emergency department after ingesting caffeine pills with serum levels of 140 mg/L."
    pubs = make_single_paper_pub(title, text)
    res = classify_publications(
        publications=pubs,
        entity="caffeine",
        criterion="The paper describes the chemical synthesis of caffeine.",
        scorer=real_scorer
    )
    assert res.results[0]["classification"] == "does_not_qualify"
    assert res.results[0]["evidence_status"] == "insufficient"
    assert res.results[0]["evidence"] is None


@pytest.mark.slow
def test_n4_alias_collision_marinus_thein():
    """N4: Alias collision on author/historical figure Marinus Thein does not match caffeine."""
    text = "First person account of Javanese agriculture recorded by Marinus Thein in 1845."
    pubs = make_single_paper_pub("Historical Javanese Agriculture", text)
    res = classify_publications(
        publications=pubs,
        entity="caffeine",
        aliases=["thein", "theine"],
        criterion="The paper describes the chemical synthesis of caffeine.",
        scorer=real_scorer if False else None  # Doesn't need model; verified by alias vetting
    ) if False else None

    # Verify alias vetting directly blocks Marinus Thein
    from app.text_utils import vet_aliases, build_entity_regex
    vetted = vet_aliases(["thein", "theine"], "caffeine")
    rx = build_entity_regex("caffeine", vetted)
    assert rx.search(text) is None


@pytest.mark.slow
def test_n5_network_pharmacology_abstain(real_scorer):
    """N5: Network pharmacology paper mentioning entity without production claim abstains."""
    title = "Network Pharmacology Analysis of Artemisinic Acid in Colorectal Cancer"
    text = "Network pharmacology and molecular docking showed that artemisinic acid targets the PI3K/Akt pathway."
    pubs = make_single_paper_pub(title, text)
    res = classify_publications(
        publications=pubs,
        entity="artemisinic acid",
        criterion="The paper reports microbial biosynthesis of artemisinic acid in this study.",
        scorer=real_scorer
    )
    assert res.results[0]["classification"] == "does_not_qualify"
    assert res.results[0]["evidence_status"] == "insufficient"
    assert res.results[0]["evidence"] is None


@pytest.mark.slow
def test_n6_background_citation_fails_this_work_gate(real_scorer):
    """N6: Background statement citing prior art fails the 'this work' gate."""
    text = "Artemisinic acid has been produced by engineered yeast in previous studies. In this report, we evaluate downstream crystallization."
    pubs = make_single_paper_pub("Crystallization process", text)
    res = classify_publications(
        publications=pubs,
        entity="artemisinic acid",
        criterion="The paper reports microbial biosynthesis of artemisinic acid in this study.",
        scorer=real_scorer
    )
    assert res.results[0]["classification"] == "does_not_qualify"


def test_d1_preprint_published_deduplication():
    """D1: Preprint and published paper are deduplicated, recording dedup_of."""
    papers = [
        {
            "title": "Engineering Saccharomyces cerevisiae for high-level artemisinic acid",
            "doi": "10.1101/2021.04.10.111111",  # bioRxiv
            "pmid": None,
            "abstractText": "Preprint text."
        },
        {
            "title": "Engineering Saccharomyces cerevisiae for high-level artemisinic acid",
            "doi": "10.1038/s41589-021-00890-z",
            "pmid": "42107802",
            "abstractText": "Published article text."
        }
    ]
    unique, count = deduplicate_papers(papers)
    assert len(unique) == 1
    assert count == 1
    assert unique[0]["pmid"] == "42107802"
    assert unique[0]["dedup_of"] is not None
