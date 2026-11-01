import pytest
import re
from typing import List, Tuple
from app.text_utils import (
    clean_text,
    build_entity_regex,
    split_sentences,
    candidate_sentences,
)
from app.scoring import (
    calculate_qualify_score,
    calculate_confidence,
    get_confidence_band,
    rank_papers,
)
from app.nli import NLIScorer
from app.classifier import classify_publications, normalise_criterion, EvidenceIntegrityError
from app.paraphrase import is_synthesis_criterion, generate_criterion_paraphrases


class MockNLIScorer(NLIScorer):
    """
    Mock scorer that returns deterministic probabilities based on pre-programmed premise rules.
    """
    def __init__(self, mapping=None, default=(0.33, 0.34, 0.33)):
        self.mapping = mapping or {}
        self.default = default

    def score(self, pairs: List[Tuple[str, str]]) -> List[Tuple[float, float, float]]:
        out = []
        for premise, hyp in pairs:
            # Check for substring match in premise or hypothesis
            matched = False
            for pattern, triple in self.mapping.items():
                if pattern in premise or pattern in hyp:
                    out.append(triple)
                    matched = True
                    break
            if not matched:
                out.append(self.default)
        return out


def test_sentence_splitting_abbreviations():
    """Verify sentence splitting on scientific abbreviations: et al., Fig. 3, E.g."""
    text = "Smith et al. described the mechanism (Fig. 3). E.g. in high yield, psilocybin was prepared."
    sents = split_sentences(text)
    assert len(sents) == 2
    assert "et al." in sents[0]
    assert "Fig. 3" in sents[0]
    assert sents[1].startswith("E.g.")


def test_word_boundary_matching():
    """Verify regex matches exact word boundaries and not inside longer tokens."""
    pat = build_entity_regex("psilocybin")
    
    # Matches valid variations
    assert pat.search("Synthesized psilocybin yesterday.") is not None
    assert pat.search("Contains psilocybins.") is not None
    assert pat.search("Psilocybin was detected.") is not None

    # Rejects substring matches inside longer words
    assert pat.search("Studied deoxypsilocybin derivatives.") is None
    assert pat.search("Identified psilocybinic acid in extracts.") is None


def test_synonym_matching():
    """Verify entity regex matches user-specified synonyms and hyphen/space variants."""
    pat = build_entity_regex("psilocybin", synonyms=["psilocybine", "4-phosphoryloxy-DMT"])
    
    assert pat.search("Derived psilocybine.") is not None
    assert pat.search("Administered 4-phosphoryloxy-DMT to subjects.") is not None
    assert pat.search("Used 4 phosphoryloxy DMT in test.") is not None


def test_confidence_symmetry():
    """Verify that confidence is symmetric for both classes."""
    tau = 0.50

    # Confident positive: m = 0.9 (far above tau=0.5)
    # Distance from tau to 1.0 is 0.5. Here (0.9 - 0.5) / 0.5 = 0.8
    # conf = 0.5 + 0.5 * 0.8 = 0.90
    conf_pos = calculate_confidence(m=0.90, tau=tau, is_qualifies=True)
    assert conf_pos >= 0.80
    assert get_confidence_band(conf_pos) == "high"

    # Confident negative: m = -0.70 (far below tau=0.5)
    # Distance from tau to -1.0 is 1.5. Here (0.5 - (-0.70)) / 1.5 = 1.2 / 1.5 = 0.8
    # conf = 0.5 + 0.5 * 0.8 = 0.90
    conf_neg = calculate_confidence(m=-0.70, tau=tau, is_qualifies=False)
    assert conf_neg >= 0.80
    assert get_confidence_band(conf_neg) == "high"

    # Boundary case: m = 0.5 (exactly at tau)
    conf_boundary = calculate_confidence(m=0.50, tau=tau, is_qualifies=True)
    assert conf_boundary == 0.50
    assert get_confidence_band(conf_boundary) == "low"


def test_evidence_rule_table_all_rows():
    """
    Verify all 4 rows of the evidence rule table:
    1. qualifies -> entailing_sentence
    2. does_not_qualify, contrast guard fired -> contrast_mention
    3. does_not_qualify, entity present, no entailment -> strongest_mention
    4. No usable text -> insufficient_text
    """
    criterion = "Psilocybin is synthesised or biosynthesised in this work"
    entity = "psilocybin"

    # Define mock mapping:
    # "total synthesis" against criterion -> ent=0.95, con=0.01 (m=0.94)
    # "total synthesis" against contrast -> ent=0.05, con=0.90
    # "administered" against criterion -> ent=0.75, con=0.05 (m=0.70 >= tau)
    # "administered" against contrast -> ent=0.98, con=0.01 (contrast guard fires!)
    # "mere mention" against criterion -> ent=0.10, con=0.80 (m=-0.70)
    # "mere mention" against contrast -> ent=0.40, con=0.40
    mock_mapping = {
        "total synthesis": (0.95, 0.04, 0.01),
        "administered": (0.75, 0.20, 0.05),
        "mere mention": (0.10, 0.10, 0.80),
    }

    class TableMockScorer(NLIScorer):
        def score(self, pairs):
            out = []
            for premise, hyp in pairs:
                is_contrast = "reagent" in hyp.lower() or "administered" in hyp.lower()
                if "total synthesis" in premise:
                    out.append((0.05, 0.05, 0.90) if is_contrast else (0.95, 0.04, 0.01))
                elif "administered" in premise:
                    out.append((0.98, 0.01, 0.01) if is_contrast else (0.75, 0.20, 0.05))
                elif "mere mention" in premise:
                    out.append((0.40, 0.20, 0.40) if is_contrast else (0.10, 0.10, 0.80))
                else:
                    out.append((0.33, 0.34, 0.33))
            return out

    scorer = TableMockScorer()

    publications = [
        # Paper 1: qualifies
        {
            "title": "We achieved total synthesis of psilocybin.",
            "abstractText": "In this paper we report total synthesis of psilocybin in high yield.",
            "pmid": "101", "doi": "10.1/101", "pmcid": None
        },
        # Paper 2: contrast guard fired
        {
            "title": "Psilocybin trial.",
            "abstractText": "Psilocybin was administered to subjects in a randomized clinical trial.",
            "pmid": "102", "doi": "10.1/102", "pmcid": None
        },
        # Paper 3: entity present, no entailment
        {
            "title": "Survey paper.",
            "abstractText": "This is a mere mention of psilocybin without any synthesis.",
            "pmid": "103", "doi": "10.1/103", "pmcid": None
        },
        # Paper 4: No usable text (no entity in title, no abstract)
        {
            "title": "Unrelated title about chemistry.",
            "abstractText": None,
            "pmid": "104", "doi": None, "pmcid": None
        }
    ]

    results, _, _, _ = classify_publications(
        publications=publications,
        entity=entity,
        criterion=criterion,
        threshold=0.50,
        contrast_delta=0.20,
        scorer=scorer
    )

    # Sort results by PMID for deterministic index inspection
    res_by_pmid = {r["pmid"]: r for r in results}

    # Row 1: qualifies -> entailing_sentence
    r1 = res_by_pmid["101"]
    assert r1["classification"] == "qualifies"
    assert r1["evidence_status"] == "entailing_sentence"
    assert "total synthesis" in r1["evidence"]
    assert r1["confidence_band"] in ["high", "medium"]

    # Row 2: does_not_qualify, contrast guard fired -> contrast_mention
    r2 = res_by_pmid["102"]
    assert r2["classification"] == "does_not_qualify"
    assert r2["evidence_status"] == "contrast_mention"
    assert "administered" in r2["evidence"]

    # Row 3: does_not_qualify, entity present, no entailment -> strongest_mention
    r3 = res_by_pmid["103"]
    assert r3["classification"] == "does_not_qualify"
    assert r3["evidence_status"] == "strongest_mention"
    assert "mere mention" in r3["evidence"]

    # Row 4: No usable text -> insufficient_text
    r4 = res_by_pmid["104"]
    assert r4["classification"] == "does_not_qualify"
    assert r4["evidence_status"] == "insufficient_text"
    assert r4["evidence"] is None
    assert r4["confidence"] is None
    assert r4["confidence_band"] == "insufficient"


def test_no_abstract_insufficient_text():
    """Verify that a paper without abstract and without entity in title becomes insufficient_text."""
    entity_regex = build_entity_regex("psilocybin")
    cands, title_has_entity, n_sents = candidate_sentences(
        title="General organic chemistry methods",
        abstract=None,
        entity_regex=entity_regex
    )
    assert cands == []
    assert not title_has_entity
    assert n_sents == 0


def test_missing_doi_pmid_handling():
    """Verify classifier handles null or missing identifiers without error."""
    class DummyScorer(NLIScorer):
        def score(self, pairs):
            return [(0.1, 0.1, 0.8) for _ in pairs]

    publications = [{
        "title": "Study mentioning psilocybin.",
        "abstractText": "We observed psilocybin properties.",
        "doi": None,
        "pmid": None,
        "pmcid": None
    }]
    results, _, _, _ = classify_publications(
        publications=publications,
        entity="psilocybin",
        criterion="Psilocybin is synthesised",
        scorer=DummyScorer()
    )
    assert len(results) == 1
    assert results[0]["doi"] is None
    assert results[0]["pmid"] is None
    assert results[0]["pmcid"] is None


def test_criterion_normalisation():
    """Verify light normalisation removes leading framing and substitutes {entity}."""
    c1 = "The paper reports that {entity} is biosynthesized."
    assert normalise_criterion(c1, "psilocybin") == "Psilocybin is biosynthesized."

    c2 = "In this work, psilocybin is isolated and synthesized."
    assert normalise_criterion(c2, "psilocybin") == "Psilocybin is isolated and synthesized."


def test_chemical_punctuation_regex():
    """Verify entity regex matches complex chemical punctuation and terms correctly."""
    pat1 = build_entity_regex("(+)-psilocybin")
    assert pat1.search("Obtained (+)-psilocybin in high purity.") is not None
    assert pat1.search("Obtained (-)-psilocybin in test.") is None

    pat2 = build_entity_regex("5-HT(2A)")
    assert pat2.search("Activation of 5-HT(2A) receptor.") is not None
    assert pat2.search("Activation of 5-HT(1A) receptor.") is None

    pat3 = build_entity_regex("N,N-DMT")
    assert pat3.search("Investigated N,N-DMT effects.") is not None
    assert pat3.search("Investigated 5-MeO-DMT effects.") is None


def test_evidence_integrity_error_on_tampering():
    """Verify EvidenceIntegrityError is raised if evidence is not a verbatim substring."""
    class TamperingScorer(NLIScorer):
        def score(self, pairs):
            return [(0.95, 0.04, 0.01) for _ in pairs]

    # Malicious or buggy paper where candidate sentences are modified
    from unittest.mock import patch
    with patch("app.classifier.candidate_sentences", return_value=(["Invented hallucinated sentence."], True, 1)):
        pub = {
            "title": "Real paper title about psilocybin",
            "abstractText": "Real abstract mentioning psilocybin.",
            "pmid": "999"
        }
        with pytest.raises(EvidenceIntegrityError):
            classify_publications(
                publications=[pub],
                entity="psilocybin",
                criterion="Psilocybin was synthesized",
                scorer=TamperingScorer()
            )


def test_criterion_conditional_contrast_guard():
    """Verify contrast guard is active for synthesis criteria and not_applicable for non-synthesis."""
    assert is_synthesis_criterion("Vanillin was synthesized or produced in this work") is True
    assert is_synthesis_criterion("Psilocybin is biosynthesised by Psilocybe cubensis") is True
    assert is_synthesis_criterion("Compound showed antifungal activity against Candida albicans") is False

    class DummyScorer(NLIScorer):
        def score(self, pairs):
            return [(0.70, 0.20, 0.10) for _ in pairs]

    pub = {
        "title": "Antifungal activity of psilocybin against yeast.",
        "abstractText": "Psilocybin showed antifungal activity against fungal strains.",
        "pmid": "antifungal_01"
    }

    res_run = classify_publications(
        publications=[pub],
        entity="psilocybin",
        criterion="Psilocybin showed antifungal activity against fungal strains",
        scorer=DummyScorer()
    )
    assert res_run.contrast_guard_active == "not_applicable"


def test_insufficient_text_null_scores():
    """Verify papers with insufficient text have null qualify_score and null confidence."""
    class DummyScorer(NLIScorer):
        def score(self, pairs):
            return [(0.5, 0.5, 0.0) for _ in pairs]

    pub = {
        "title": "Unrelated title about quantum mechanics",
        "abstractText": None,
        "pmid": "no_text_01"
    }
    results, _, _, _ = classify_publications(
        publications=[pub],
        entity="psilocybin",
        criterion="Psilocybin was synthesized",
        scorer=DummyScorer()
    )
    assert results[0]["evidence_status"] == "insufficient_text"
    assert results[0]["qualify_score"] is None
    assert results[0]["confidence"] is None
    assert results[0]["confidence_band"] == "insufficient"


