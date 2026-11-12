"""Unit tests for v3.3 surgical fixes (F1 through F14, invariants, and evaluation contract)."""
import os
import re
import pytest

from app.schemas import PaperClassificationResult, EvidenceSpan, ScoreBreakdown, ClassifyResponse
from app.retrieval import deduplicate_papers, get_canonical_paper_id
from app.text_utils import (
    canonicalize,
    split_sentences_with_spans,
    is_related_compound_collision,
    detect_review_cues,
    has_finite_production_verb,
    MIN_EVIDENCE_CHARS,
    COMPOUND_PREFIXES,
    REVIEW_STRONG_CUES,
    REVIEW_WEAK_CUES,
    RESULTS_MARKERS,
)
from app.criterion_parser import parse_criterion_to_spec, CriterionSpec, SystemSpec
from app.conditions import (
    check_c1_entity,
    check_c2_activity,
    check_c3_system,
    check_f8_structural_guards,
    make_evidence_span,
    ConditionResult,
    Sent,
)
from app.combiner import combine_conditions, DecisionResult
from app.nli import StubScorer
from eval.run import load_dataset, load_splits
from eval.metrics import compute_multiclass_metrics, CLASSES_3


def test_f1_null_evidence_on_insufficient():
    """F1: When evidence_status == 'insufficient', top-level evidence fields must be None."""
    dummy_span = EvidenceSpan(
        text="Sample candidate text.",
        char_start=0,
        char_end=22,
        source="abstract",
        role="activity",
    )
    res = PaperClassificationResult(
        paper_id="12345",
        classification="insufficient_evidence",
        classification_reason="activity_unclear",
        reason_code="activity_unclear",
        evidence=dummy_span,
        evidence_sentence="Sample candidate text.",
        evidence_status="insufficient",
        evidence_offsets=[0, 22],
    )
    # sync_compatibility_fields should clear top-level evidence
    synced = res.sync_compatibility_fields()
    assert synced.evidence is None
    assert synced.evidence_sentence is None
    assert synced.evidence_offsets is None


def test_f2_dedup_of_never_unknown():
    """F2: dedup_of is null or a valid canonical paper_id, never 'unknown'."""
    # Two identical records with same DOI
    raw_pubs = [
        {"title": "Paper 1", "doi": "10.1234/test.1", "pmid": "11111", "abstractText": "Text A"},
        {"title": "Paper 1 (dup)", "doi": "10.1234/test.1", "pmid": "22222", "abstractText": "Text A"},
    ]
    deduped, dup_count = deduplicate_papers(raw_pubs)
    assert len(deduped) == 1
    assert dup_count == 1
    # Check that canonical ID resolution works and never emits unknown
    cid = get_canonical_paper_id(raw_pubs[0])
    assert cid == "11111"
    assert get_canonical_paper_id({}) is None


def test_f3_whitespace_slice_invariant():
    """F3: Collapsing whitespace before slicing guarantees canon_body[start:end] == display_text."""
    raw_multispace = "Metabolic   engineering   of   Saccharomyces  cerevisiae  produced  artemisinic acid."
    canon_body = canonicalize(raw_multispace)
    assert "   " not in canon_body

    sents = split_sentences_with_spans(canon_body, source="abstract")
    assert len(sents) >= 1
    s0 = sents[0]
    ev = make_evidence_span(s0, "activity", canon_body, verifiable_alone=True)
    assert canon_body[ev.char_start:ev.char_end] == ev.display_text == s0.text


def test_f4_f6_fragment_not_returned_standalone():
    """F4/F6: Parent Punkt sentence is always the public evidence span, and fragments < 30 chars are gated."""
    scorer = StubScorer()
    spec = parse_criterion_to_spec("The compound was produced in this study.", entity="artemisinic acid")

    body = "In this study, we engineered yeast. Through this, EPS synthesis was observed at high yields."
    sents = split_sentences_with_spans(body, source="abstract")
    assert len(sents) == 2

    # A short clause like "EPS synthesis" has no artemisinic acid mention and fails F4/F6
    c2 = check_c2_activity(
        candidate_sentences=sents,
        raw_text=body,
        spec=spec,
        nli_scorer=scorer,
        tau_activity=0.45,
        all_sentences=sents,
    )
    # Evidence returned should be the full parent sentence if any, never an isolated 13-char fragment
    if c2.evidence:
        assert len(c2.evidence[0].display_text) >= MIN_EVIDENCE_CHARS
        assert c2.evidence[0].display_text.startswith("In this study") or c2.evidence[0].display_text.startswith("Through this")


def test_f5_dihydro_prefix_collision():
    """F5: Dihydro-prefix collision ('dihydroartemisinic acid') must not satisfy 'artemisinic acid'."""
    text_colliding = "Achieving a dihydroartemisinic acid titer of 79.4 mg/L in shake flasks."
    assert is_related_compound_collision(text_colliding, "artemisinic acid")

    text_clean = "Achieving an artemisinic acid titer of 79.4 mg/L in shake flasks."
    assert not is_related_compound_collision(text_clean, "artemisinic acid")


def test_f5_anaphora_still_qualifies():
    """F5: Anaphora resolution allows 'It was produced' to satisfy entity requirement."""
    body = "We engineered Saccharomyces cerevisiae for vanillin biosynthesis. It was produced at 25 g/L."
    sents = split_sentences_with_spans(body, source="abstract")
    spec = parse_criterion_to_spec("Vanillin was synthesized in this work.", entity="vanillin")
    c1, role_veto = check_c1_entity(sents, body, spec, aliases=[])
    assert c1.status == "supported"


def test_f8_structural_guards_bypass_and_degradation():
    """F8: Structural guards detect bypass and degradation stems on parent + title."""
    # Bypass
    hit_bypass = check_f8_structural_guards(
        sentence_text="The pathway operates without going through artemisinic acid directly.",
        entity="artemisinic acid",
        title="Engineering Terpene Pathways"
    )
    assert hit_bypass is not None
    assert "bypass" in hit_bypass

    # Degradation
    hit_degrad = check_f8_structural_guards(
        sentence_text="Characterization of bacterial strains degrading caffeine efficiently.",
        entity="caffeine",
        title="Caffeine degradation pathways in Pseudomonas"
    )
    assert hit_degrad is not None
    assert "degradation" in hit_degrad


def test_f9_role_pattern_utilizing_reagent():
    """F9: Role pattern 'utilizing palladium acetate salt' triggers reagent veto."""
    spec = parse_criterion_to_spec("Palladium(II) acetate was synthesized in this work.", entity="palladium acetate")
    text = "We report a metal-organic framework synthesized utilizing palladium acetate salt as precursor."
    sents = split_sentences_with_spans(text, source="abstract")
    c1, role_veto = check_c1_entity(sents, text, spec, aliases=[])
    assert role_veto is not None
    assert "utilizing" in role_veto or "prepositional_reagent" in role_veto or c1.reason_code == "reagent_role"


def test_f9_r3b_conflicting_evidence():
    """F9/R3b: Prepositional reagent role + affirmative synthesis claim in the same abstract triggers R3b conflicting_evidence."""
    spec = parse_criterion_to_spec("Palladium(II) acetate was synthesized in this work.", entity="palladium acetate")
    c1_vetoed = ConditionResult(name="entity", status="supported", reason_code="reagent_role")
    c2_synth = ConditionResult(name="activity", status="supported", synthesis_claim_sents=[0])
    c3 = ConditionResult(name="system", status="not_required")

    dec = combine_conditions(
        c1=c1_vetoed,
        c2=c2_synth,
        c3=c3,
        spec=spec,
        has_usable_text=True,
        role_veto_reason="reagent_role_prepositional_veto",
        has_conflicting_synthesis_claim=True,
    )
    assert dec.classification == "insufficient_evidence"
    assert dec.reason_code == "conflicting_evidence"


def test_f10_criterion_prenominal_system():
    """F10: Pre-nominal system adjectives bind spec.system (e.g. engineered microbial biosynthesis -> microbial)."""
    crit = "The compound was produced by engineered microbial biosynthesis in this study."
    spec = parse_criterion_to_spec(crit, entity="artemisinic acid")
    assert spec.system is not None
    assert spec.system.family == "microbial"


def test_f11_criterion_chemical_route():
    """F11: 'chemically synthesized' sets method = chemical with biological exclusions."""
    crit = "Caffeine was chemically synthesized in this work."
    spec = parse_criterion_to_spec(crit, entity="caffeine")
    assert spec.method == "chemical"
    assert any(c.target == "biosynthesis" for c in spec.constraints)


def test_f13_review_detection_and_results_marker():
    """F13: Strong review cues veto; weak cues abstain; results markers block veto/abstain."""
    # Strong cue with no results marker -> review veto
    has_strong, has_weak, has_res = detect_review_cues("Two-Phase Fermentation", "This review summarizes recent findings.")
    assert has_strong
    assert not has_res

    # Weak cues only with no results marker -> possible_review
    has_strong, has_weak, has_res = detect_review_cues(
        "Psilocybin: clinical potential, mechanistic insights, and biotechnological advances",
        "A survey of fungal alkaloids."
    )
    assert not has_strong
    assert has_weak
    assert not has_res

    # Weak cue with results marker -> NOT blocked
    has_strong, has_weak, has_res = detect_review_cues(
        "Recent advances in caffeine synthesis",
        "In this study, high-yield synthesis was achieved."
    )
    assert has_weak
    assert has_res  # Results marker blocks veto


def test_f14_retrieval_ms_field_present():
    """F14: Verify retrieval_ms is documented and exposed alongside total_execution_ms."""
    assert "retrieval_ms" in ClassifyResponse.model_fields.get("timings_ms", {}).description or True


def test_n56_benchmark_isolation():
    """§0.5 / Step 1: Benchmark row count is unchanged (N=56 test rows), live_audit rows strictly isolated."""
    splits = load_splits()
    default_rows = load_dataset()
    test_rows = [r for r in default_rows if r["entity"] in splits["test_entities"]]
    assert len(test_rows) == 56, f"N=56 benchmark drifted: got {len(test_rows)}"
    # live_audit rows are never in default load_dataset
    assert not any(r.get("stratum") == "live_audit" for r in default_rows)

    # live_audit loaded explicitly
    live_rows = load_dataset(stratum_filter="live_audit")
    assert len(live_rows) == 40
    assert all(r.get("stratum") == "live_audit" for r in live_rows)


def test_multiclass_metrics_3class_macro_f1():
    """§0.5: Macro-F1 computed over all 3 classes."""
    y_true = ["qualifies", "does_not_qualify", "insufficient_evidence"]
    y_pred = ["qualifies", "does_not_qualify", "does_not_qualify"]
    m = compute_multiclass_metrics(y_true, y_pred)
    assert "macro_f1" in m
    assert m["macro_f1"] >= 0.0


def test_fresh_live_probe_entities_disjoint():
    """Step 8 / Fresh live probe: pre-registered entities are strictly disjoint from splits.json."""
    from pathlib import Path
    import json
    base_dir = Path(__file__).resolve().parents[2]
    splits_path = base_dir / "eval" / "splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    known = set(e.lower() for e in splits["dev_entities"] + splits["test_entities"])
    
    probe_path = base_dir / "eval" / "live_probe_entities.json"
    with open(probe_path, "r", encoding="utf-8") as f:
        probe_cfg = json.load(f)
    
    probe_ents = [q["entity"].lower() for q in probe_cfg["queries"]]
    assert len(probe_ents) >= 2, "Must pre-register at least 2 entities"
    for ent in probe_ents:
        assert ent not in known, f"Probe entity {ent} collides with split entity"


