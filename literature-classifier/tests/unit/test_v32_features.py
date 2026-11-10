"""Comprehensive Tier 1 Unit & Feature Tests for v3.2 Reconciliations (WS12–WS17)."""
import itertools
import pytest
from pydantic import ValidationError

from app.combiner import combine_conditions, combine, CombinerInput, RULES
from app.conditions import (
    ConditionResult,
    check_c1_entity,
    check_c2_activity,
    check_c3_system,
)
from app.criterion_parser import (
    CriterionSpec,
    SystemSpec,
    ClauseConstraint,
    parse_criterion_to_spec,
)
from app.nli import StubScorer
from app.reasons import REASON_REGISTRY, get_reason_info
from app.schemas import PaperClassificationResult, ConfidenceDetails
from app.text_utils import (
    Sent,
    split_sentences_with_spans,
    split_clauses_bracket_aware,
    resolve_anaphora,
    build_entity_regex,
)
from eval.calibrate import fit_logistic_mle


def test_ws17_r2_registry_authoritative():
    """WS17-R2: mention_only and background_citation are evidence-based negatives; unstated_threshold is #18."""
    assert len(REASON_REGISTRY) == 18
    assert get_reason_info("mention_only").label == "does_not_qualify"
    assert get_reason_info("background_citation").label == "does_not_qualify"
    assert get_reason_info("unstated_threshold").label == "insufficient_evidence"
    assert get_reason_info("unstated_threshold").kind == "abstain"


def test_ws17_r5_truth_table_total_and_allow_partial():
    """WS17-R5: All 27 C1xC2xC3 states terminate cleanly + allow-partial precedence rows."""
    assert len(RULES) == 12
    spec_sys = CriterionSpec(
        entity="vanillin",
        entity_role="product",
        activity="synthesis",
        system=SystemSpec(name="yeast", family="yeast"),
    )
    for s1, s2, s3 in itertools.product(["supported", "contradicted", "unknown"], repeat=3):
        c1 = ConditionResult(name="entity", status=s1, score=1.0 if s1 == "supported" else -1.0)
        c2 = ConditionResult(name="activity", status=s2, score=1.0 if s2 == "supported" else -1.0)
        c3 = ConditionResult(name="system", status=s3, score=1.0 if s3 == "supported" else -1.0)
        dec = combine_conditions(c1, c2, c3, spec_sys)
        assert dec.classification in ("qualifies", "does_not_qualify", "insufficient_evidence")

    # Mandatory allow-partial precedence rows:
    spec_partial = CriterionSpec(
        entity="vanillin",
        entity_role="product",
        activity="synthesis",
        system=SystemSpec(name="yeast", family="yeast"),
        unparsed_clauses=["with dose-dependence"],
    )
    c_sup = ConditionResult(name="entity", status="supported", score=1.0)
    c_act_sup = ConditionResult(name="activity", status="supported", score=0.9)
    c_bg = ConditionResult(name="activity", status="contradicted", score=-0.5, reason_code="background_citation")
    c_ws = ConditionResult(name="system", status="contradicted", score=-0.9, reason_code="wrong_system")

    # allow-partial AND wrong_system -> does_not_qualify(wrong_system)
    d_ws = combine_conditions(c_sup, c_act_sup, c_ws, spec_partial)
    assert d_ws.classification == "does_not_qualify" and d_ws.reason_code == "wrong_system"

    # allow-partial AND background_citation -> does_not_qualify(background_citation)
    d_bg = combine_conditions(c_sup, c_bg, c_sup, spec_partial)
    assert d_bg.classification == "does_not_qualify" and d_bg.reason_code == "background_citation"

    # allow-partial AND all-supported -> insufficient_evidence(unsupported_criterion_clause)
    d_part = combine_conditions(c_sup, c_act_sup, c_sup, spec_partial)
    assert d_part.classification == "insufficient_evidence" and d_part.reason_code == "unsupported_criterion_clause"


def test_ws14_a_conflicting_evidence_routing():
    """WS14-A: R3a vetoes reagent-only papers; R3b routes conflicting synthesis + reagent role to conflicting_evidence."""
    spec = CriterionSpec(entity="vanillin", entity_role="product", activity="synthesis")
    c_veto = ConditionResult(name="entity", status="contradicted", score=-0.95, reason_code="reagent_role")
    c_act = ConditionResult(name="activity", status="supported", score=0.85)
    c_sys = ConditionResult(name="system", status="not_required", score=1.0)

    # R3a: reagent-only (no conflicting affirmative synthesis) -> veto
    d_veto = combine_conditions(
        c_veto, c_act, c_sys, spec,
        role_veto_reason="role_pattern: in the presence of vanillin",
        has_conflicting_synthesis_claim=False,
    )
    assert d_veto.classification == "does_not_qualify" and d_veto.reason_code == "reagent_role"

    # R3b: reagent phrase + affirmative synthesis claim elsewhere -> conflicting_evidence
    d_conf = combine_conditions(
        c_veto, c_act, c_sys, spec,
        role_veto_reason="role_pattern: in the presence of vanillin",
        has_conflicting_synthesis_claim=True,
    )
    assert d_conf.classification == "insufficient_evidence" and d_conf.reason_code == "conflicting_evidence"


def test_ws15_b_bracket_depth_iupac_and_trailing_negation():
    """WS15-B / WS17-R5: Bracket-aware clause splitting preserves IUPAC names cleanly and isolates trailing negation."""
    canon = (
        "Using 2,2'-bipyridine, N,N-dimethyltryptamine, and 6-trifluoromethyldibenzo[b,d]oxepine as ligands, "
        "we evaluated catalytic cross-coupling under aqueous conditions at 80 C, but vanillin was not synthesized in this work."
    )
    sents = split_sentences_with_spans(canon)
    assert len(sents) == 1
    clauses = split_clauses_bracket_aware(sents[0], min_len=100)
    joined = " ".join(c.text for c in clauses)
    # All three IUPAC names must remain intact (never split on internal commas like 2,2' or N,N or [b,d])
    assert any("2,2'-bipyridine" in c.text for c in clauses)
    assert any("N,N-dimethyltryptamine" in c.text for c in clauses)
    assert any("6-trifluoromethyldibenzo[b,d]oxepine" in c.text for c in clauses)
    # Trailing contrast/negation clause is isolated cleanly
    assert any("but vanillin was not synthesized in this work" in c.text for c in clauses)
    # Byte-exact offset invariant per clause
    for c in clauses:
        assert canon[c.start:c.end] == c.text


def test_ws14_b_c_comparative_organism_in_vitro_and_d6():
    """WS14-B & WS14-C (D6): Comparative organisms don't pass C3; in_vitro stays separate; abstract mismatch -> wrong_system."""
    text = "Unlike Escherichia coli systems previously reported, Saccharomyces cerevisiae produced vanillin in batch culture."
    sents = split_sentences_with_spans(text)

    spec_ecoli = CriterionSpec(
        entity="vanillin", entity_role="product", activity="biosynthesis",
        system=SystemSpec(name="escherichia coli", family="bacteria"),
    )
    c3_ecoli = check_c3_system(sents, text, spec_ecoli, is_fulltext=False)
    # "Unlike Escherichia coli..." must NOT satisfy E. coli, and active S. cerevisiae yields wrong_system (D6)
    assert c3_ecoli.status == "contradicted"
    assert c3_ecoli.reason_code == "wrong_system"
    assert len(c3_ecoli.comparative_mentions) == 1

    # in_vitro separation
    spec_iv = parse_criterion_to_spec("The paper reports chemoenzymatic synthesis of vanillin in vitro in this study.", entity="vanillin")
    assert spec_iv.system is not None
    assert spec_iv.system.family == "in_vitro"

    text_yeast = "We produced vanillin in engineered yeast cells."
    sents_y = split_sentences_with_spans(text_yeast)
    c3_iv = check_c3_system(sents_y, text_yeast, spec_iv, is_fulltext=False)
    assert c3_iv.status == "contradicted" and c3_iv.reason_code == "wrong_system"


def test_ws14_d_g3_exclusions_and_numeric_thresholds():
    """WS14-D: G3 grammar parses exclusions, numeric thresholds, and compound clauses without 422."""
    c1 = "The paper reports biosynthesis of vanillin excluding chemical synthesis in this study."
    s1 = parse_criterion_to_spec(c1, entity="vanillin")
    assert len(s1.constraints) == 1 and s1.constraints[0].kind == "exclusion"
    assert "chemical synthesis" in s1.constraints[0].target

    c2 = "The paper reports microbial biosynthesis of vanillin with yield > 1 g/L in this study."
    s2 = parse_criterion_to_spec(c2, entity="vanillin")
    assert len(s2.constraints) == 1 and s2.constraints[0].kind == "numeric_threshold"
    assert s2.constraints[0].value == 1.0 and s2.constraints[0].unit == "g/L"

    c3 = "The paper reports biosynthesis of vanillin with conversion > 80% and excluding chemical synthesis in this study."
    s3 = parse_criterion_to_spec(c3, entity="vanillin")
    assert len(s3.constraints) == 2 and s3.unparsed_clauses == []

    # Unit normalization: 500 mg/L == 0.5 g/L
    c4 = "The paper reports biosynthesis of vanillin with titer >= 500 mg/L in this study."
    s4 = parse_criterion_to_spec(c4, entity="vanillin")
    assert s4.constraints[0].value == 0.5 and s4.constraints[0].unit == "g/L"


def test_ws14_e_cross_sentence_anaphora_gating():
    """WS14-E: Resolve cross-sentence anaphora within ANAPHORA_WINDOW=2 with no lookahead."""
    text = (
        "Vanillin was identified in fermented extracts. "
        "This compound was produced at 8.2 g/L by engineered yeast."
    )
    sents = split_sentences_with_spans(text)
    reg = build_entity_regex("vanillin")
    amap = resolve_anaphora(sents, reg, window=2)
    assert amap == {1: 0}


def test_ws15_a_structural_title_isolation():
    """WS15-A: Empty title vs synthesis-keyword title produces identical classifications; title never leaks into evidence."""
    from app.classifier import classify_publications
    scorer = StubScorer(default_margin=-0.4)
    pub_with_title = {
        "pmid": "111",
        "title": "De novo synthesis and production of caffeine in engineered yeast",
        "abstractText": "Participants drank coffee and blood pressure was monitored.",
    }
    pub_empty_title = {
        "pmid": "111",
        "title": "",
        "abstractText": "Participants drank coffee and blood pressure was monitored.",
    }
    r1 = classify_publications([pub_with_title], "caffeine", "The paper describes synthesis of caffeine.", scorer=scorer, pipeline="v3")
    r2 = classify_publications([pub_empty_title], "caffeine", "The paper describes synthesis of caffeine.", scorer=scorer, pipeline="v3")
    assert r1.results[0]["classification"] == r2.results[0]["classification"]
    assert r1.results[0]["evidence_sentence"] == r2.results[0]["evidence_sentence"]


def test_ws17_r4_d5_schema_validator_and_mle_fit():
    """WS17-R4 & WS12-A: D5 schema rejects numeric confidence on vetoes/abstentions/categorical negatives; MLE fits logistic."""
    # MLE recovery check
    margins = [-0.6, -0.3, -0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    a, b, brier = fit_logistic_mle(margins, labels)
    assert a > 0.5 and brier < 0.15

    # D5 rejects numeric confidence on insufficient_evidence
    with pytest.raises(ValidationError):
        PaperClassificationResult(
            paper_id="1", classification="insufficient_evidence",
            reason_code="activity_unclear", evidence_status="insufficient",
            confidence=ConfidenceDetails(score=0.4, value=0.4, band="Low"),
        )

    # D5 rejects numeric confidence on mention_only / background_citation / review_article / reagent_role
    for cat_code in ("mention_only", "background_citation", "review_article", "reagent_role"):
        with pytest.raises(ValidationError):
            PaperClassificationResult(
                paper_id="1", classification="does_not_qualify",
                reason_code=cat_code, evidence_status="contradicted",
                confidence=ConfidenceDetails(score=0.85, value=0.85, band="High"),
            )

    # D5 ALLOWS calibrated confidence on different_activity and wrong_system
    ok_res = PaperClassificationResult(
        paper_id="1", classification="does_not_qualify",
        reason_code="different_activity", evidence_status="contradicted",
        confidence=ConfidenceDetails(score=0.88, value=0.88, band="High"),
    )
    assert ok_res.confidence_details.score == 0.88
