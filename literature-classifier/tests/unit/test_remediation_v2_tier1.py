"""Tier 1 Unit tests for v2 remediation: alias vetting, role patterns, this-work gate, deduplication, and monotone confidence."""
import pytest
from app.text_utils import (
    vet_aliases,
    clean_markup_and_unescape,
    canonicalize,
    detect_this_work_cues,
    is_review_article,
    has_production_verb,
    build_entity_regex,
    Sent,
    verify,
)
from app.hypotheses import check_role_patterns, detect_intent
from app.retrieval import deduplicate_papers, normalize_title_for_dedupe
from app.scoring import score_to_calibrated_prob, assign_confidence_band


def test_alias_vetting_denylist():
    """Verify alias vetting rejects denylisted collisions, surnames, and short tokens."""
    raw_aliases = [
        "thein",
        "Theine",
        "Marinus Thein",
        "mat",
        "aid",
        "lead",
        "acid",
        "salt",
        "base",
        "yield",
        "4-hydroxy-3-methoxybenzaldehyde",  # Valid chemical alias
        "vanillic aldehyde",                # Valid chemical alias
        "caffeine"                          # Same as entity
    ]
    vetted = vet_aliases(raw_aliases, entity="caffeine")
    assert "thein" not in [v.lower() for v in vetted]
    assert "theine" not in [v.lower() for v in vetted]
    assert "marinus thein" not in [v.lower() for v in vetted]
    assert "mat" not in vetted
    assert "aid" not in vetted
    assert "caffeine" not in vetted
    assert "4-hydroxy-3-methoxybenzaldehyde" in vetted
    assert "vanillic aldehyde" in vetted


def test_role_patterns_detection():
    """Verify role regex patterns detect catalyst, reagent, and precursor usage."""
    sent_catalyst = "In the presence of palladium(II) acetate, dibenzoxepines were prepared in 88% yield."
    sent_by = "The coupling was catalyzed by Pd(OAc)2 under microwave irradiation."
    sent_using = "Using Pd(OAc)2 as catalyst, biaryls were afforded."
    sent_product = "Palladium(II) acetate was synthesized from metallic palladium and acetic acid."

    reason_cat = check_role_patterns(sent_catalyst, "palladium(II) acetate", ["Pd(OAc)2"])
    assert reason_cat is not None
    assert "in the presence of" in reason_cat.lower()

    reason_by = check_role_patterns(sent_by, "palladium(II) acetate", ["Pd(OAc)2"])
    assert reason_by is not None
    assert "catalyzed by" in reason_by.lower()

    reason_using = check_role_patterns(sent_using, "palladium(II) acetate", ["Pd(OAc)2"])
    assert reason_using is not None

    reason_prod = check_role_patterns(sent_product, "palladium(II) acetate", ["Pd(OAc)2"])
    assert reason_prod is None  # Pure synthesis sentence does NOT trigger role pattern


def test_this_work_gate():
    """Verify detection of experimental positive cues vs background citation cues."""
    sent_positive = "In this study, we synthesized vanillin using an engineered Escherichia coli strain."
    sent_background = "Artemisinic acid has been produced by engineered yeast in previous studies."
    sent_neutral = "Vanillin is an important aromatic aldehyde used in the flavor industry."

    pos_p, bg_p = detect_this_work_cues(sent_positive)
    assert pos_p is True
    assert bg_p is False

    pos_b, bg_b = detect_this_work_cues(sent_background)
    assert pos_b is False
    assert bg_b is True

    pos_n, bg_n = detect_this_work_cues(sent_neutral)
    assert pos_n is False
    assert bg_n is False


def test_review_and_production_verb_detection():
    """Verify review pre-filter and production verb detection."""
    review_title = "A Comprehensive Review of Natural Vanillin Production"
    review_text = "Here, we synthesize current knowledge on the metabolic pathways of phenolic compounds."
    regular_text = "Under optimized fermenter conditions, 4.5 g/L vanillin was yielded."

    assert is_review_article(review_title, regular_text) is True
    assert is_review_article("Natural Bio-flavor", review_text) is True
    assert is_review_article("Microbial engineering", regular_text) is False

    assert has_production_verb("Vanillin was obtained in 82% yield.") is True
    assert has_production_verb("Vanillin was purchased from Sigma-Aldrich.") is False
    assert has_production_verb("We evaluated the cytotoxic activity of curcumin.") is False


def test_deduplication_and_markup_cleaning():
    """Verify preprint deduplication and HTML markup cleaning."""
    dirty_text = "<i>E. coli</i> was incubated with 10<sup>-3</sup> M glucose &amp; galactose."
    cleaned = clean_markup_and_unescape(dirty_text)
    assert cleaned == "E. coli was incubated with 10-3 M glucose & galactose."

    papers = [
        {
            "title": "Biosynthesis of artemisinic acid in engineered <i>Saccharomyces cerevisiae</i>",
            "doi": "10.1101/2021.05.01.123456",  # bioRxiv DOI
            "pmid": None,
            "abstractText": "Preprint version of yeast engineering."
        },
        {
            "title": "Biosynthesis of artemisinic acid in engineered Saccharomyces cerevisiae",
            "doi": "10.1038/s41589-021-00890-z", # Nature journal DOI
            "pmid": "42107802",
            "abstractText": "Final peer-reviewed version of yeast engineering."
        }
    ]

    unique_papers, dedup_count = deduplicate_papers(papers)
    assert len(unique_papers) == 1
    assert dedup_count == 1
    # Peer-reviewed journal paper should be retained
    assert unique_papers[0]["pmid"] == "42107802"
    assert unique_papers[0]["dedup_of"] is not None


def test_confidence_band_strict_monotonicity():
    """Verify that calibrated probability strictly maps monotonically to confidence bands."""
    band_order = {"low": 0, "medium": 1, "high": 2}

    prev_rank = -1
    for i in range(101):
        p = i / 100.0
        band = assign_confidence_band(p)
        rank = band_order[band]
        assert rank >= prev_rank, f"Monotonicity violation at P={p}: {band} followed lower rank {prev_rank}"
        prev_rank = rank


def test_intent_detection():
    """Verify criterion intent classification into synthesis, pharmacology, or other."""
    assert detect_intent("The paper describes the synthesis or production of vanillin.") == "synthesis"
    assert detect_intent("Microbial biosynthesis of artemisinic acid in this study.") == "synthesis"
    assert detect_intent("Anti-inflammatory activity of curcumin in mice.") == "pharmacology"
    assert detect_intent("The paper reports crystal structure determination.") == "other"

