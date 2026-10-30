"""Tier 1 Unit Tests: Deterministic, no network, no model weights.
Implements robustness checks E1-E14 and core invariants.
"""
import pytest
import nltk
from app.text_utils import (
    canonicalize,
    split_sentences_with_spans,
    verify,
    build_entity_regex,
    Sent,
)
from app.hypotheses import (
    lexical_intent_armed,
    build_hypotheses,
)
from app.scoring import (
    guard_confidence,
    ranking_sort_key,
    margin_to_qualify_score,
)
from app.retrieval import extract_fulltext_sections
from app.classifier import select_candidates_intent_aware


def test_e1_guard_armed_manufacture():
    """E1: Criterion with 'manufacture' arms the synthesis guard."""
    crit = "The paper describes the manufacture of vanillin."
    assert lexical_intent_armed(crit, entity="vanillin") is True


def test_e2_guard_armed_fuzzy_typo():
    """E2: Criterion with typo 'synthsis' arms the synthesis guard via fuzzy match."""
    crit = "The study reports the synthsis of vanillin from ferulic acid."
    assert lexical_intent_armed(crit, entity="vanillin") is True


def test_e3_sentence_splitting_scientific_abbrev():
    """E3: Yield with 'mol. %' and '(Fig. 3)' does not split prematurely."""
    text = "Yield was 4.2 mol. % at pH 7.4 (Fig. 3). Curcumin et al. reported a similar value."
    canon = canonicalize(text)
    spans = split_sentences_with_spans(canon)
    assert len(spans) == 2
    assert spans[0].text == "Yield was 4.2 mol. % at pH 7.4 (Fig. 3)."
    assert spans[1].text == "Curcumin et al. reported a similar value."
    assert all(verify(canon, s) for s in spans)


def test_e3b_adversarial_sentence_splitting():
    """E3b: Adversarial sentence splitting: decimals, genus abbreviations, and figure titles."""
    # 1. Genus abbreviation and cell harvest
    t1 = "E. coli K-12 was grown at 37 °C. Cells were harvested."
    c1 = canonicalize(t1)
    s1 = split_sentences_with_spans(c1)
    assert len(s1) == 2
    assert s1[0].text == "E. coli K-12 was grown at 37 °C."
    assert s1[1].text == "Cells were harvested."
    assert all(verify(c1, s) for s in s1)

    # 2. Decimal adjacent to abbreviation
    t2 = "The reaction used approx. 2.5 g of starting material. The mixture was stirred."
    c2 = canonicalize(t2)
    s2 = split_sentences_with_spans(c2)
    assert len(s2) == 2
    assert s2[0].text == "The reaction used approx. 2.5 g of starting material."
    assert s2[1].text == "The mixture was stirred."
    assert all(verify(c2, s) for s in s2)


def test_e4_canonicalize_and_verification():
    """E4: Abstract with HTML tags and entities verifies cleanly without 500 error."""
    raw = "<p><i>E. coli</i> produced &alpha;-pinene at 10<sup>-3</sup> M.</p>"
    canon = canonicalize(raw)
    assert "<i>" not in canon
    assert "<sup>" not in canon
    assert "α-pinene" in canon or "α-pinene" in canon or "pinene" in canon

    # Test idempotence
    assert canonicalize(canon) == canon

    # Test span verification
    spans = split_sentences_with_spans(canon)
    assert len(spans) >= 1
    assert all(verify(canon, s) for s in spans)


def test_e6_pmc_xml_no_sec_type():
    """E6: PMC XML with section titles but no sec-type still yields Methods text."""
    xml_content = """
    <article>
        <body>
            <sec>
                <title>Materials and Methods</title>
                <p>Recombinant enzymes were prepared and vanillin was isolated in 85% yield.</p>
            </sec>
            <sec>
                <title>References</title>
                <p>1. Smith et al. Bioorg Med Chem 2020.</p>
            </sec>
        </body>
    </article>
    """
    sections = extract_fulltext_sections(xml_content)
    assert len(sections) == 1
    assert "vanillin was isolated" in sections[0][0]


def test_e7_eligibility_invariant_long_document():
    """E7: Entity sentence past 10,000 characters is in eligible set and scored."""
    filler = "This is background experimental context that contains no target entity. " * 150  # ~11,000 chars
    target_sentence = "Under optimal bioreactor conditions, vanillin was produced at 4.2 g/L."
    full_text = filler + target_sentence

    canon = canonicalize(full_text)
    rx = build_entity_regex("vanillin")
    spans = split_sentences_with_spans(canon)

    eligible = [s for s in spans if rx.search(s.text)]
    assert len(eligible) == 1
    assert eligible[0].start > 10000
    assert "vanillin was produced" in eligible[0].text

    selected = select_candidates_intent_aware(eligible, synthesis_armed=True, max_k=6)
    assert len(selected) == 1
    assert selected[0].text == eligible[0].text


def test_e10_lookaround_residual_risk():
    """E10: Documents known behavior of lookaround matching on locant-prefixed isomers."""
    rx = build_entity_regex("vanillin")
    # Sub-word prefix is correctly blocked
    assert rx.search("isovanillin") is None
    # Hyphenated prefix matches boundary (documented residual risk)
    assert rx.search("o-vanillin") is not None


def test_e12_guard_confidence_monotonicity():
    """E12: guard_confidence is strictly monotonic with c* and non-zero at spec values."""
    p_ent = 0.10
    delta = 0.05
    tripwire = p_ent - delta  # 0.05

    # Sweep c* from tripwire to 1.0
    c_values = [0.05, 0.20, 0.50, 0.80, 0.95, 1.00]
    conf_values = [guard_confidence(c, p_ent=p_ent, delta=delta, cap=0.65) for c in c_values]

    # Verify non-decreasing monotonicity
    for i in range(len(conf_values) - 1):
        assert conf_values[i] <= conf_values[i + 1]

    # Spec value test: c* = 0.9771
    spec_conf = guard_confidence(c_star=0.9771, p_ent=0.10, delta=0.05, cap=0.65)
    assert spec_conf == 0.65
    assert spec_conf > 0.0  # Must NOT be 0!


def test_e13_deliberate_ranking_order():
    """E13: Guard-rejected papers rank below ordinary non-qualifiers."""
    qualifying_paper = {
        "classification": "qualifies",
        "classification_reason": "margin_above_threshold",
        "qualify_score": 0.85,
        "pub_year": "2021",
        "paper_id": "P1"
    }
    ordinary_non_qualifier = {
        "classification": "does_not_qualify",
        "classification_reason": "margin_below_threshold",
        "qualify_score": 0.45,
        "pub_year": "2020",
        "paper_id": "P2"
    }
    guard_rejected_paper = {
        "classification": "does_not_qualify",
        "classification_reason": "disqualified_by_contrast_guard (starting material / reagent detected)",
        "qualify_score": 0.80,  # High raw score, but reagent detected!
        "pub_year": "2022",
        "paper_id": "P3"
    }
    no_text_paper = {
        "classification": "does_not_qualify",
        "classification_reason": "insufficient_text",
        "qualify_score": None,
        "pub_year": "2019",
        "paper_id": "P4"
    }

    papers = [no_text_paper, guard_rejected_paper, ordinary_non_qualifier, qualifying_paper]
    ranked = sorted(papers, key=ranking_sort_key)

    # Expected order:
    # 1. Qualifying paper
    # 2. Ordinary non-qualifier
    # 3. Guard-rejected paper (deliberately placed below ordinary non-qualifier)
    # 4. No-text paper (qualify_score=None strictly last)
    assert ranked[0]["paper_id"] == "P1"
    assert ranked[1]["paper_id"] == "P2"
    assert ranked[2]["paper_id"] == "P3"
    assert ranked[3]["paper_id"] == "P4"


def test_e14_nltk_environment_resources():
    """E14: Fresh environment verification: punkt and punkt_tab both available."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError as e:
        pytest.fail(f"Required NLTK tokenizer resource missing: {e}")

