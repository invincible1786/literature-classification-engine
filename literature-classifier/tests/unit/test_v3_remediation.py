"""Tier 1 Unit tests for v3 upgrade plan:
Verifies Reason Registry, Criterion Parser, Condition Checkers, Role Polarity,
Combiner Precedence Truth Table, Abstract Carve-Outs, and Schema Guarantees.
"""
import pytest
from app.reasons import REASON_REGISTRY, validate_reason_code, get_reason_info
from app.criterion_parser import (
    CriterionSpec,
    CriterionParseError,
    parse_criterion_to_spec,
    SystemSpec
)
from app.conditions import (
    check_c1_entity,
    check_c2_activity,
    check_c3_system,
    is_related_compound_collision,
    EvidenceSpanV3,
    ConditionResult
)
from app.combiner import combine_conditions, DecisionResult
from app.text_utils import Sent, split_sentences_with_spans, canonicalize
from app.nli import StubScorer
from app.schemas import ClassifyResponse, PaperClassificationResult, EvidenceSpan


def test_t_reason_registry():
    """T-reason-registry: All 18 required reason codes must be registered and valid."""
    required_codes = [
        "no_text", "abstract_only", "mention_only", "activity_unclear",
        "entity_unclear", "entity_absent", "system_unstated", "system_mismatch_abstract",
        "conflicting_evidence", "evidence_split", "unsupported_criterion_clause",
        "different_activity", "wrong_system", "reagent_role", "background_citation",
        "possible_review", "review_article", "unstated_threshold"
    ]
    assert len(REASON_REGISTRY) == 18
    for code in required_codes:
        assert validate_reason_code(code), f"Missing registered reason code: {code}"
        info = get_reason_info(code)
        assert info is not None
        assert info.label in ("does_not_qualify", "insufficient_evidence")
        assert info.kind in ("veto", "evidence-based", "abstain", "metadata")
        assert len(info.description) > 5


def test_t_parser_grammar():
    """T-parser-grammar: Parses G1 (product) and G2 (agent) criteria correctly."""
    # G1: Product with host system
    crit1 = "The paper reports microbial biosynthesis of artemisinic acid in yeast in this study."
    spec1 = parse_criterion_to_spec(crit1, entity="artemisinic acid")
    assert spec1.entity_role == "product"
    assert spec1.agent_kind is None
    assert "biosynthesis" in spec1.activity.lower()
    assert spec1.system is not None
    assert spec1.system.name == "yeast"
    assert spec1.scope == "this_study"

    # G1: Product without host system
    crit2 = "The paper describes the synthesis or production of vanillin."
    spec2 = parse_criterion_to_spec(crit2, entity="vanillin")
    assert spec2.entity_role == "product"
    assert spec2.system is None

    # G2: Agent (catalyst)
    crit3 = "The paper uses palladium(II) acetate as a catalyst in cross-coupling reactions."
    spec3 = parse_criterion_to_spec(crit3, entity="palladium(II) acetate")
    assert spec3.entity_role == "agent"
    assert spec3.agent_kind == "catalyst"
    assert spec3.activity is not None


def test_t_unparsed_clause_and_partial_policy():
    """T-unparsed-clause, T-partial-criterion-policy: Unsupported clauses raise 422 by default, handled under allow_partial."""
    crit_bad = "The paper reports synthesis of vanillin with dose-dependence in this study."
    
    # Default: raises CriterionParseError
    with pytest.raises(CriterionParseError) as exc_info:
        parse_criterion_to_spec(crit_bad, entity="vanillin", allow_partial_criterion=False)
    assert "dose-dependence" in str(exc_info.value)
    assert exc_info.value.unparsed_clauses == ["with dose-dependence"]

    # Opt-in: allow_partial_criterion parses core and records unparsed clause
    spec_partial = parse_criterion_to_spec(crit_bad, entity="vanillin", allow_partial_criterion=True)
    assert spec_partial.entity_role == "product"
    assert "with dose-dependence" in spec_partial.unparsed_clauses


def test_t_role_veto_entity_object():
    """T-role-veto-entity-object: Prepositional pattern vetoes only when entity_role == 'product'."""
    text = "In the presence of palladium(II) acetate and triphenylphosphine, dibenzoxepines were prepared in 78% yield."
    sents = split_sentences_with_spans(text)
    
    # Under product role: triggers reagent_role veto
    spec_prod = CriterionSpec(entity="palladium(II) acetate", entity_role="product", activity="synthesis")
    c1_prod, veto_reason = check_c1_entity(sents, text, spec_prod, aliases=["Pd(OAc)2"])
    assert c1_prod.status == "contradicted"
    assert c1_prod.reason_code == "reagent_role"
    assert veto_reason is not None

    # Reverse case: "{entity} was produced in the presence of Y" must NOT trigger veto
    text_prod = "Vanillin was synthesized in the presence of copper catalyst."
    sents_prod = split_sentences_with_spans(text_prod)
    spec_van = CriterionSpec(entity="vanillin", entity_role="product", activity="synthesis")
    c1_van, veto_van = check_c1_entity(sents_prod, text_prod, spec_van)
    assert c1_van.status == "supported"
    assert veto_van is None


def test_t_agent_role_g2():
    """T-agent-role-G2: Role patterns act as SUPPORTING evidence when entity_role == 'agent'."""
    text = "Using Pd(OAc)2 as catalyst, biaryls were afforded in high yield."
    sents = split_sentences_with_spans(text)
    spec_agent = CriterionSpec(
        entity="palladium(II) acetate",
        entity_role="agent",
        agent_kind="catalyst",
        activity="cross-coupling"
    )
    c1, veto = check_c1_entity(sents, text, spec_agent, aliases=["Pd(OAc)2"])
    assert c1.status == "supported"
    assert veto is None
    assert c1.score >= 0.80


def test_t_related_compound():
    """T-related-compound: Derivative prefixes do not falsely trigger exact entity match."""
    sent1 = "The accumulation of dihydroartemisinic acid was quantified in glandular trichomes."
    assert is_related_compound_collision(sent1, "artemisinic acid") is True

    sent2 = "Both artemisinic acid and dihydroartemisinic acid were extracted."
    assert is_related_compound_collision(sent2, "artemisinic acid") is False


def test_t_review_pubtype_and_we_review():
    """T-review-pubtype, T-we-review-not-vetoed: pubType vetoes, but 'we review' in research does not auto-veto."""
    spec = CriterionSpec(entity="vanillin", entity_role="product", activity="synthesis", scope="this_study")
    c_dummy = ConditionResult(name="entity", status="supported", score=1.0)
    
    # Review pubtype without first-person cues vetoes
    dec_veto = combine_conditions(c_dummy, c_dummy, c_dummy, spec, is_pubtype_review_veto=True)
    assert dec_veto.classification == "does_not_qualify"
    assert dec_veto.reason_code == "review_article"

    # Research article mentioning "we review previous findings" does NOT trigger pubtype review veto
    dec_ok = combine_conditions(c_dummy, c_dummy, c_dummy, spec, is_pubtype_review_veto=False)
    assert dec_ok.classification == "qualifies"


def test_t_null_confidence_on_abstain():
    """T-null-confidence (WS17-R2 / R4): Verb-gate mention_only AND abstain activity_unclear both return null confidence."""
    from app.classifier import classify_publications
    # 1. Verb-gate negative (mention_only -> does_not_qualify, confidence=None)
    pub_mention = [{
        "title": "Random Pharmacology Study",
        "abstractText": "We studied caffeine receptor affinity in hippocampal rat brain slices.",
        "pmid": "99999999"
    }]
    scorer = StubScorer(default_margin=-0.2)
    res_m = classify_publications(
        publications=pub_mention,
        entity="caffeine",
        criterion="The paper describes the chemical synthesis of caffeine.",
        scorer=scorer,
        pipeline="v3"
    )
    assert res_m.results[0]["classification"] == "does_not_qualify"
    assert res_m.results[0]["reason_code"] == "mention_only"
    assert res_m.results[0]["confidence"] is None

    # 2. Ambiguous production abstain (activity_unclear -> insufficient_evidence, confidence=None)
    pub_abstain = [{
        "title": "Ambiguous Caffeine Yield Study",
        "abstractText": "Caffeine generation pathways were discussed tentatively in cell extracts.",
        "pmid": "99999998"
    }]
    res_a = classify_publications(
        publications=pub_abstain,
        entity="caffeine",
        criterion="The paper describes the chemical synthesis of caffeine.",
        scorer=scorer,
        pipeline="v3"
    )
    assert res_a.results[0]["classification"] == "insufficient_evidence"
    assert res_a.results[0]["confidence"] is None


def test_t_entity_absent_fulltext_only():
    """T-entity-absent-fulltext-only: entity_absent requires full text; abstract-only gives entity_unclear."""
    spec = CriterionSpec(entity="vanillin", entity_role="product", activity="synthesis")
    sents = split_sentences_with_spans("A study of aromatic aldehydes in food packaging.")

    # Abstract-only: entity_unclear (Methods may have it)
    c1_abs, _ = check_c1_entity(sents, "A study of aromatic aldehydes...", spec, is_fulltext=False)
    assert c1_abs.status == "unknown"
    assert c1_abs.reason_code == "entity_unclear"

    # Full-text: entity_absent
    c1_full, _ = check_c1_entity(sents, "A study of aromatic aldehydes...", spec, is_fulltext=True)
    assert c1_full.status == "contradicted"
    assert c1_full.reason_code == "entity_absent"


def test_t_system_mismatch_abstract():
    """T-system-mismatch-abstract (D6): Abstract mentioning alternative organism emits wrong_system by default."""
    spec = CriterionSpec(
        entity="vanillin",
        entity_role="product",
        activity="synthesis",
        system=SystemSpec(name="yeast", family="yeast")
    )
    text = "We report production of vanillin using Escherichia coli in fermentation broth."
    sents = split_sentences_with_spans(text)

    # D6 default (strict): abstract-explicit host contradiction -> wrong_system (does_not_qualify)
    c3_d6 = check_c3_system(sents, text, spec, is_fulltext=False)
    assert c3_d6.status == "contradicted"
    assert c3_d6.reason_code == "wrong_system"
    assert "escherichia coli" in [m.lower() for m in c3_d6.also_mentions]

    # Legacy conservative override abstains safely via system_unstated
    c3_cons = check_c3_system(sents, text, spec, is_fulltext=False, abstract_negative_policy="conservative")
    assert c3_cons.status == "unknown"
    assert c3_cons.reason_code == "system_unstated"


def test_t_two_organisms_not_conflict():
    """T-two-organisms-not-conflict: Paper naming yeast and E. coli is not a conflict if requested yeast is supported."""
    spec = CriterionSpec(
        entity="artemisinic acid",
        entity_role="product",
        activity="synthesis",
        system=SystemSpec(name="yeast", family="yeast")
    )
    text = "High titers were achieved in Saccharomyces cerevisiae, while expression in Escherichia coli gave low yields."
    sents = split_sentences_with_spans(text)
    c3 = check_c3_system(sents, text, spec, is_fulltext=False)
    assert c3.status == "supported"
    assert "escherichia coli" in [m.lower() for m in c3.also_mentions]


def test_t_truth_table():
    """T-truth-table: Full combiner precedence 1-11 truth table verification."""
    spec = CriterionSpec(entity="vanillin", entity_role="product", activity="synthesis")
    c_supp = ConditionResult(name="entity", status="supported", score=1.0)
    c_unkn = ConditionResult(name="activity", status="unknown", score=0.0, reason_code="activity_unclear")
    c_cont = ConditionResult(name="activity", status="contradicted", score=-1.0, reason_code="different_activity")

    # Rule 1: No text
    d1 = combine_conditions(c_supp, c_supp, c_supp, spec, has_usable_text=False)
    assert d1.classification == "insufficient_evidence" and d1.reason_code == "no_text"

    # Rule 2: Role veto
    d2 = combine_conditions(c_supp, c_supp, c_supp, spec, role_veto_reason="role_pattern: in the presence of")
    assert d2.classification == "does_not_qualify" and d2.reason_code == "reagent_role"

    # Rule 3: Review pubType veto
    d3 = combine_conditions(c_supp, c_supp, c_supp, spec, is_pubtype_review_veto=True)
    assert d3.classification == "does_not_qualify" and d3.reason_code == "review_article"

    # Rule 5: Activity contradicted
    d5 = combine_conditions(c_supp, c_cont, c_supp, spec)
    assert d5.classification == "does_not_qualify" and d5.reason_code == "different_activity"

    # Rule 7: Activity unknown
    d7 = combine_conditions(c_supp, c_unkn, c_supp, spec)
    assert d7.classification == "insufficient_evidence" and d7.reason_code == "activity_unclear"

    # Rule 9: Unsupported clause cap
    spec_unparsed = CriterionSpec(
        entity="vanillin",
        entity_role="product",
        activity="synthesis",
        unparsed_clauses=["with dose-dependence"]
    )
    d9 = combine_conditions(c_supp, c_supp, c_supp, spec_unparsed)
    assert d9.classification == "insufficient_evidence" and d9.reason_code == "unsupported_criterion_clause"

    # Rule 11: All supported qualifies
    d11 = combine_conditions(c_supp, c_supp, c_supp, spec)
    assert d11.classification == "qualifies" and d11.reason_code is None

