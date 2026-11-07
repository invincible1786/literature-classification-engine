"""Classification engine executing candidate extraction, intent gating, NLI inference, role guard, and sentence coupling."""
import hashlib
import logging
import math
import re
import statistics
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Literal

from app.nli import NLIScorer, StubScorer
from app.text_utils import (
    canonicalize,
    split_sentences_with_spans,
    verify,
    build_entity_regex,
    vet_aliases,
    has_production_verb,
    is_review_article,
    detect_this_work_cues,
    compute_sha256,
    clean_markup_and_unescape,
    Sent,
)
from app.hypotheses import (
    detect_intent,
    lexical_intent_armed,
    build_hypotheses,
    load_contrast_hypotheses,
    get_negative_role_hypothesis,
    check_role_patterns,
    SYNTH_STEMS,
)
from app.scoring import (
    margin_to_qualify_score,
    score_to_calibrated_prob,
    calculate_confidence,
    assign_confidence_band,
    ranking_sort_key,
    CONFIDENCE_KIND,
)
from app.schemas import EvidenceSpan, ConfidenceDetails, ScoreBreakdown
from app.criterion_parser import (
    CriterionSpec,
    CriterionParseError,
    parse_criterion_to_spec,
)
from app.conditions import (
    EvidenceSpanV3,
    ConditionResult,
    check_c1_entity,
    check_c2_activity,
    check_c3_system,
)
from app.combiner import combine_conditions, DecisionResult
from app.reasons import validate_reason_code

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.45
DEFAULT_CONTRAST_DELTA = 0.00
DEFAULT_MAX_CANDIDATES = 6


class EvidenceIntegrityError(Exception):
    """Raised if strict verbatim verification fails and strict mode is requested."""
    pass


def normalise_criterion(criterion: str) -> str:
    """Normalize criterion whitespace."""
    return " ".join(criterion.split()) if criterion else ""


@dataclass
class ScoredSentence:
    sent: Sent
    positive_margin: float
    role_score: float
    final_score: float
    p_ent: float
    gate_flags: List[str]
    guard_reason: Optional[str] = None


class ClassificationRunResult(tuple):
    """
    Backwards-compatible tuple subclass providing both legacy tuple unpacking:
    (results, normalised_criterion, tau, delta)
    and structured attributes:
    .hypotheses_used, .contrast_guard_active, .guard_arming, .intent, .criterion_spec
    """
    def __new__(
        cls,
        results: List[Dict[str, Any]],
        normalised_criterion: str,
        tau: float,
        delta: float,
        hypotheses_used: Optional[List[str]] = None,
        contrast_guard_active: bool = False,
        guard_arming: Optional[Dict[str, Any]] = None,
        intent: str = "synthesis",
        criterion_spec: Optional[CriterionSpec] = None,
        timings_ms: Optional[Dict[str, float]] = None
    ):
        instance = super().__new__(cls, (results, normalised_criterion, tau, delta))
        instance.results = results
        instance.normalised_criterion = normalised_criterion
        instance.tau = tau
        instance.delta = delta
        instance.hypotheses_used = hypotheses_used or []
        instance.contrast_guard_active = contrast_guard_active
        instance.guard_arming = guard_arming or {}
        instance.intent = intent
        instance.criterion_spec = criterion_spec
        instance.timings_ms = timings_ms or {}
        return instance


def select_candidates_intent_aware(
    eligible_sentences: List[Sent],
    intent: str = "synthesis",
    synthesis_armed: Optional[bool] = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_k: Optional[int] = None
) -> List[Sent]:
    """Select candidate sentences using intent and verb gating."""
    if max_k is not None:
        max_candidates = max_k
    if not eligible_sentences:
        return []

    is_synth = synthesis_armed if synthesis_armed is not None else (intent == "synthesis")
    if not is_synth:
        return eligible_sentences[:max_candidates]

    prod_sents = [s for s in eligible_sentences if has_production_verb(s.text)]
    return prod_sents[:max_candidates] if prod_sents else []


def _classify_publications_v2(
    publications: List[Dict[str, Any]],
    entity: str,
    criterion: str,
    scorer: NLIScorer,
    tau: float = DEFAULT_THRESHOLD,
    delta: float = DEFAULT_CONTRAST_DELTA,
    aliases: Optional[List[str]] = None,
    custom_contrast_hypotheses: Optional[List[str]] = None,
    user_paraphrases: Optional[List[str]] = None,
    two_hit_rule: bool = False,
    strict_verification: bool = False
) -> ClassificationRunResult:
    """Legacy v2 pipeline: sentence-level NLI, role pattern guard, Platt calibration."""
    vetted_aliases = vet_aliases(aliases, entity)
    intent = detect_intent(criterion)
    synthesis_armed = lexical_intent_armed(criterion)

    hypotheses = build_hypotheses(
        criterion=criterion,
        entity=entity,
        synthesis_armed=synthesis_armed,
        user_paraphrases=user_paraphrases
    )
    n_pos_hyps = len(hypotheses)
    h_negative_role = get_negative_role_hypothesis(entity)

    contrast_hypotheses = (
        custom_contrast_hypotheses
        if custom_contrast_hypotheses is not None
        else load_contrast_hypotheses(entity, vetted_aliases)
    )

    guard_arming_meta = {
        "synthesis_armed": synthesis_armed,
        "intent_detected": intent,
        "n_contrast_hypotheses": len(contrast_hypotheses),
        "two_hit_rule": two_hit_rule
    }

    all_pairs: List[Tuple[str, str]] = []
    paper_metadata: List[Dict[str, Any]] = []
    paper_pair_spans: List[Optional[Tuple[int, int]]] = []

    stride = n_pos_hyps + 1

    for pub in publications:
        raw_title = pub.get("title")
        raw_abstract = pub.get("abstractText")
        title = clean_markup_and_unescape(raw_title) if raw_title else None
        abstract = clean_markup_and_unescape(raw_abstract) if raw_abstract else None

        canon_title = canonicalize(title)
        canon_abstract = canonicalize(abstract)

        combined_text = f"{canon_title}. {canon_abstract}".strip() if canon_title else canon_abstract
        canon_doc = combined_text

        if not canon_doc:
            paper_metadata.append({"status": "no_text", "pub": pub, "canon": "", "doc_sha": None})
            paper_pair_spans.append(None)
            continue

        doc_sha = compute_sha256(canon_doc)

        if is_review_article(canon_title, canon_abstract):
            paper_metadata.append({
                "status": "review_abstain",
                "pub": pub,
                "canon": canon_doc,
                "doc_sha": doc_sha,
                "k_eligible": 0
            })
            paper_pair_spans.append(None)
            continue

        all_sentences = split_sentences_with_spans(canon_doc)
        entity_regex = build_entity_regex(entity, vetted_aliases)
        eligible = [s for s in all_sentences if entity_regex.search(s.text)]
        k_eligible = len(eligible)

        scored_candidates = select_candidates_intent_aware(
            eligible,
            intent=intent,
            synthesis_armed=synthesis_armed,
            max_candidates=DEFAULT_MAX_CANDIDATES
        )

        if not scored_candidates:
            paper_metadata.append({
                "status": "insufficient",
                "pub": pub,
                "canon": canon_doc,
                "doc_sha": doc_sha,
                "k_eligible": k_eligible
            })
            paper_pair_spans.append(None)
            continue

        start_idx = len(all_pairs)
        for s in scored_candidates:
            for h in hypotheses:
                all_pairs.append((s.text, h))
            all_pairs.append((s.text, h_negative_role))
        end_idx = len(all_pairs)

        paper_metadata.append({
            "status": "ready",
            "pub": pub,
            "canon": canon_doc,
            "doc_sha": doc_sha,
            "k_eligible": k_eligible,
            "k_scored": len(scored_candidates),
            "scored_sents": scored_candidates
        })
        paper_pair_spans.append((start_idx, end_idx))

    all_nli_triples = scorer.score(all_pairs) if all_pairs else []
    results: List[Dict[str, Any]] = []

    for idx, meta in enumerate(paper_metadata):
        pub = meta["pub"]
        status = meta["status"]
        canon_doc = meta["canon"]
        doc_sha = meta["doc_sha"]
        paper_id = str(pub.get("pmid") or pub.get("doi") or pub.get("title") or f"paper_{idx}")
        title = pub.get("title")
        doi = pub.get("doi")
        pmid = pub.get("pmid")
        pmcid = pub.get("pmcid")
        pub_year = pub.get("pubYear")
        dedup_of = pub.get("dedup_of")
        k_eligible = meta.get("k_eligible", 0)
        k_scored = meta.get("k_scored", 0)

        if status == "no_text":
            results.append({
                "paper_id": paper_id,
                "title": title,
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "pub_year": pub_year,
                "classification": "does_not_qualify",
                "classification_reason": "insufficient_text (no abstract or full-text available)",
                "evidence": None,
                "evidence_sentence": "No abstract or full text available",
                "evidence_status": "no_text",
                "evidence_source": "abstract",
                "evidence_offsets": None,
                "text_sha256": None,
                "qualify_score": None,
                "confidence": None,
                "confidence_band": "Low",
                "confidence_details": ConfidenceDetails(score=None, band="Low"),
                "confidence_kind": CONFIDENCE_KIND,
                "score_breakdown": ScoreBreakdown(positive_margin=None, role_score=None, final_score=None, gate_flags=["no_text"]),
                "guard_reason": None,
                "dedup_of": dedup_of,
                "k_eligible": 0,
                "k_scored": 0,
                "reagent_signal": 0.0,
            })
            continue

        if status == "review_abstain":
            results.append({
                "paper_id": paper_id,
                "title": title,
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "pub_year": pub_year,
                "classification": "does_not_qualify",
                "classification_reason": "disqualified_as_review_article",
                "evidence": None,
                "evidence_sentence": "No sufficient evidence found in abstract (review article)",
                "evidence_status": "insufficient",
                "evidence_source": "abstract",
                "evidence_offsets": None,
                "text_sha256": doc_sha,
                "qualify_score": 0.15,
                "confidence": 0.15,
                "confidence_band": "Low",
                "confidence_details": ConfidenceDetails(score=0.15, band="Low"),
                "confidence_kind": CONFIDENCE_KIND,
                "score_breakdown": ScoreBreakdown(positive_margin=None, role_score=None, final_score=None, gate_flags=["review_article"]),
                "guard_reason": "review_article_abstain",
                "dedup_of": dedup_of,
                "k_eligible": k_eligible,
                "k_scored": 0,
                "reagent_signal": 0.0,
            })
            continue

        if status == "insufficient":
            results.append({
                "paper_id": paper_id,
                "title": title,
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "pub_year": pub_year,
                "classification": "does_not_qualify",
                "classification_reason": "insufficient_evidence (no candidate sentence with production verb)",
                "evidence": None,
                "evidence_sentence": "No sufficient evidence found in abstract",
                "evidence_status": "insufficient",
                "evidence_source": "abstract",
                "evidence_offsets": None,
                "text_sha256": doc_sha,
                "qualify_score": 0.12,
                "confidence": 0.12,
                "confidence_band": "Low",
                "confidence_details": ConfidenceDetails(score=0.12, band="Low"),
                "confidence_kind": CONFIDENCE_KIND,
                "score_breakdown": ScoreBreakdown(positive_margin=None, role_score=None, final_score=None, gate_flags=["no_production_verb"]),
                "guard_reason": None,
                "dedup_of": dedup_of,
                "k_eligible": k_eligible,
                "k_scored": 0,
                "reagent_signal": 0.0,
            })
            continue

        span = paper_pair_spans[idx]
        assert span is not None
        start_idx, end_idx = span
        paper_triples = all_nli_triples[start_idx:end_idx]
        scored_sents = meta["scored_sents"]

        sentence_evaluations: List[ScoredSentence] = []

        for s_idx, sent_obj in enumerate(scored_sents):
            c_offset = s_idx * stride
            cand_pos_triples = paper_triples[c_offset : c_offset + n_pos_hyps]
            cand_role_triple = paper_triples[c_offset + n_pos_hyps]

            cand_margins = [t[0] - t[2] for t in cand_pos_triples]
            cand_p_ents = [t[0] for t in cand_pos_triples]

            pos_margin = max(cand_margins)
            p_ent = max(cand_p_ents)
            role_score = cand_role_triple[0]

            gate_flags: List[str] = []
            has_pos_cue, has_bg_cue = detect_this_work_cues(sent_obj.text)
            margin_penalty = 0.0
            if has_bg_cue and not has_pos_cue:
                margin_penalty += 0.60
                gate_flags.append("this_work_background")
            elif has_pos_cue:
                gate_flags.append("this_work_positive")

            entity_regex = build_entity_regex(entity, vetted_aliases)
            role_pattern_match = check_role_patterns(sent_obj.text, entity, vetted_aliases)
            if role_pattern_match:
                gate_flags.append("role_pattern_match")

            final_score = pos_margin - role_score - margin_penalty
            if has_bg_cue and not has_pos_cue:
                final_score = min(final_score, tau - 0.10)

            sentence_evaluations.append(ScoredSentence(
                sent=sent_obj,
                positive_margin=round(pos_margin, 4),
                role_score=round(role_score, 4),
                final_score=round(final_score, 4),
                p_ent=round(p_ent, 4),
                gate_flags=gate_flags,
                guard_reason=role_pattern_match
            ))

        best_sent_eval = max(sentence_evaluations, key=lambda s: s.final_score)
        winning_sent = best_sent_eval.sent

        assert verify(canon_doc, winning_sent), "Evidence text must match canonical text at character offsets!"
        assert winning_sent.text == canon_doc[winning_sent.start:winning_sent.end]

        sorted_evals = sorted(sentence_evaluations, key=lambda s: s.final_score, reverse=True)
        runner_up_final = sorted_evals[1].final_score if len(sorted_evals) > 1 else None

        contrast_pairs = [(winning_sent.text, ch) for ch in contrast_hypotheses]
        contrast_triples = scorer.score(contrast_pairs) if contrast_pairs else []
        c_star = max((t[0] for t in contrast_triples), default=0.0)

        guard_triggered = False
        guard_reason = best_sent_eval.guard_reason

        if synthesis_armed:
            if best_sent_eval.guard_reason:
                guard_triggered = True
            elif c_star >= (best_sent_eval.p_ent - delta):
                guard_triggered = True
                guard_reason = "contrast_tripwire_exceeded (starting material / reagent detected)"

        single_borderline = False
        if two_hit_rule and (tau <= best_sent_eval.final_score < (tau + 0.10)):
            has_second_hit = any(
                idx != 0 and s.final_score >= (tau - 0.15)
                for idx, s in enumerate(sorted_evals)
            )
            if not has_second_hit:
                single_borderline = True

        if guard_triggered:
            decision = "does_not_qualify"
            reason = f"disqualified_by_guard ({guard_reason})"
            evidence_status = "contradicted"
        elif single_borderline:
            decision = "does_not_qualify"
            reason = "single_borderline_sentence (two-hit rule triggered)"
            evidence_status = "contradicted"
        elif best_sent_eval.final_score >= tau:
            decision = "qualifies"
            reason = "margin_above_threshold"
            evidence_status = "supported"
        else:
            decision = "does_not_qualify"
            reason = "margin_below_threshold"
            evidence_status = "contradicted"

        evidence_source = "fulltext" if winning_sent.start >= len(canon_doc) else "abstract"
        evidence_obj = EvidenceSpan(
            text=winning_sent.text,
            char_start=winning_sent.start,
            char_end=winning_sent.end,
            source=evidence_source
        )

        conf_score = calculate_confidence(
            m_star=best_sent_eval.final_score,
            tau=tau,
            decision=decision,
            guard_triggered=guard_triggered,
            c_star=c_star,
            p_ent=best_sent_eval.p_ent,
            delta=delta
        )
        conf_band = assign_confidence_band(
            conf_score,
            best_margin=best_sent_eval.final_score,
            runner_up_margin=runner_up_final
        )
        q_score = margin_to_qualify_score(best_sent_eval.final_score)

        score_breakdown = ScoreBreakdown(
            positive_margin=best_sent_eval.positive_margin,
            role_score=best_sent_eval.role_score,
            final_score=best_sent_eval.final_score,
            gate_flags=best_sent_eval.gate_flags
        )

        results.append({
            "paper_id": paper_id,
            "title": title,
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "pub_year": pub_year,
            "classification": decision,
            "classification_reason": reason,
            "evidence": evidence_obj,
            "evidence_sentence": winning_sent.text,
            "evidence_status": evidence_status,
            "evidence_source": evidence_source,
            "evidence_offsets": [winning_sent.start, winning_sent.end],
            "text_sha256": doc_sha,
            "qualify_score": q_score,
            "confidence": round(conf_score or 0.0, 4),
            "confidence_band": conf_band.capitalize(),
            "confidence_details": ConfidenceDetails(score=round(conf_score or 0.0, 4), band=conf_band.lower()),
            "confidence_kind": CONFIDENCE_KIND,
            "score_breakdown": score_breakdown,
            "guard_reason": guard_reason,
            "dedup_of": dedup_of,
            "k_eligible": k_eligible,
            "k_scored": k_scored,
            "reagent_signal": round(c_star, 4),
        })

    ranked_results = sorted(results, key=ranking_sort_key)
    for rank_idx, r in enumerate(ranked_results, start=1):
        r["rank"] = rank_idx

    return ClassificationRunResult(
        results=ranked_results,
        normalised_criterion=criterion.strip(),
        tau=tau,
        delta=delta,
        hypotheses_used=hypotheses,
        contrast_guard_active=synthesis_armed,
        guard_arming=guard_arming_meta,
        intent=intent
    )


def classify_publications(
    publications: List[Dict[str, Any]],
    entity: str,
    criterion: str,
    scorer: Optional[NLIScorer] = None,
    tau: float = DEFAULT_THRESHOLD,
    delta: float = DEFAULT_CONTRAST_DELTA,
    aliases: Optional[List[str]] = None,
    expand_aliases: bool = True,
    custom_contrast_hypotheses: Optional[List[str]] = None,
    user_paraphrases: Optional[List[str]] = None,
    two_hit_rule: bool = False,
    strict_verification: bool = False,
    pipeline: str = "v3",
    abstract_negative_policy: str = "conservative",
    allow_partial_criterion: bool = False,
    criterion_spec: Optional[CriterionSpec] = None
) -> ClassificationRunResult:
    """
    Main classification orchestrator supporting both v3 (multi-condition 3-class)
    and v2 (legacy sentence-level binary) pipelines.
    """
    if scorer is None:
        scorer = StubScorer()

    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    if pipeline == "v2":
        return _classify_publications_v2(
            publications=publications,
            entity=entity,
            criterion=criterion,
            scorer=scorer,
            tau=tau,
            delta=delta,
            aliases=aliases,
            custom_contrast_hypotheses=custom_contrast_hypotheses,
            user_paraphrases=user_paraphrases,
            two_hit_rule=two_hit_rule,
            strict_verification=strict_verification
        )

    # ==========================
    # Phase 3: Criterion Parsing
    # ==========================
    t_parse_start = time.perf_counter()
    if criterion_spec is not None:
        spec = criterion_spec
    else:
        spec = parse_criterion_to_spec(
            criterion=criterion,
            entity=entity,
            allow_partial_criterion=allow_partial_criterion
        )
    timings["criterion_parsing_ms"] = (time.perf_counter() - t_parse_start) * 1000.0

    vetted_aliases = vet_aliases(aliases, entity)
    intent = detect_intent(criterion)

    results: List[Dict[str, Any]] = []

    # ==========================
    # Per-Paper Evaluation Loop
    # ==========================
    t_eval_start = time.perf_counter()
    for idx, pub in enumerate(publications):
        raw_title = pub.get("title")
        raw_abstract = pub.get("abstractText")
        title = clean_markup_and_unescape(raw_title) if raw_title else None
        abstract = clean_markup_and_unescape(raw_abstract) if raw_abstract else None
        raw_text = pub.get("raw_text") or f"{raw_title or ''} {raw_abstract or ''}".strip()
        doc_sha = pub.get("source_record_sha256") or compute_sha256(raw_text)

        canon_title = canonicalize(title)
        canon_abstract = canonicalize(abstract)
        combined_text = f"{canon_title}. {canon_abstract}".strip() if canon_title else canon_abstract
        canon_doc = combined_text

        paper_id = str(pub.get("pmid") or pub.get("doi") or pub.get("title") or f"paper_{idx}")
        doi = pub.get("doi")
        pmid = pub.get("pmid")
        pmcid = pub.get("pmcid")
        pub_year = pub.get("pubYear")
        pub_type = pub.get("pubType", "")
        dedup_of = pub.get("dedup_of")
        retrieved_by = pub.get("retrieved_by", "intent")
        is_fulltext = bool(pub.get("fulltext_sections"))

        # Precedence Rule 1: No usable text
        if not canon_doc:
            results.append({
                "api_version": "v3",
                "paper_id": paper_id,
                "title": title,
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "pub_year": pub_year,
                "classification": "insufficient_evidence",
                "classification_reason": "no_text",
                "reason_code": "no_text",
                "criterion_spec": spec,
                "evidence": None,
                "evidence_sentence": None,
                "evidence_status": "no_text",
                "evidence_source": "abstract_only",
                "evidence_offsets": None,
                "source_record_sha256": doc_sha,
                "text_sha256": None,
                "aliases_used": vetted_aliases,
                "retrieved_by": retrieved_by,
                "conditions": {},
                "qualify_score": None,
                "confidence": None,
                "confidence_band": "Low",
                "confidence_details": ConfidenceDetails(score=None, band="Low", method="not_fitted"),
                "confidence_kind": CONFIDENCE_KIND,
                "score_breakdown": ScoreBreakdown(positive_margin=None, role_score=None, final_score=None, gate_flags=["no_text"]),
                "guard_reason": None,
                "dedup_of": dedup_of,
                "k_eligible": 0,
                "k_scored": 0,
                "reagent_signal": 0.0,
                "ranking_score": -10.0,
                "timings_ms": {}
            })
            continue

        # Split sentences
        sentences = split_sentences_with_spans(canon_doc)

        # Check pubType review veto
        has_pos_exp = any(detect_this_work_cues(s.text)[0] for s in sentences)
        is_review = is_review_article(canon_title, canon_abstract) or "review" in pub_type.lower()
        is_pubtype_review_veto = is_review and not has_pos_exp and spec.scope == "this_study"

        # Check C1: Entity
        c1, role_veto_reason = check_c1_entity(
            sentences=sentences,
            raw_text=raw_text,
            spec=spec,
            aliases=vetted_aliases,
            is_fulltext=is_fulltext
        )

        # Soft candidate selection for activity
        entity_regex = build_entity_regex(entity, vetted_aliases)
        eligible_sentences = [s for s in sentences if entity_regex.search(s.text)]
        candidate_sentences = select_candidates_intent_aware(
            eligible_sentences,
            intent=intent,
            max_candidates=DEFAULT_MAX_CANDIDATES
        )

        # Check C2: Activity
        c2 = check_c2_activity(
            candidate_sentences=candidate_sentences,
            raw_text=raw_text,
            spec=spec,
            nli_scorer=scorer,
            tau_activity=tau,
            tau_review=0.60
        )

        # Check C3: System
        c3 = check_c3_system(
            sentences=sentences,
            raw_text=raw_text,
            spec=spec,
            is_fulltext=is_fulltext,
            abstract_negative_policy=abstract_negative_policy  # type: ignore
        )

        # Combine via strict 11 precedence rules
        dec: DecisionResult = combine_conditions(
            c1=c1,
            c2=c2,
            c3=c3,
            spec=spec,
            has_usable_text=True,
            role_veto_reason=role_veto_reason,
            is_pubtype_review_veto=is_pubtype_review_veto,
            is_evidence_split=False
        )

        # Winning evidence processing & invariant assertion
        winning_ev_obj = None
        evidence_text = None
        evidence_offsets = None
        if dec.primary_evidence is not None:
            winning_ev_obj = dec.primary_evidence.to_v2_span()
            evidence_text = dec.primary_evidence.display_text
            evidence_offsets = [dec.primary_evidence.raw_start, dec.primary_evidence.raw_end]

            # Invariant check: ensure display_text is present in canonical document
            if strict_verification:
                assert evidence_text in canon_doc or dec.primary_evidence.raw_text in raw_text, (
                    "Evidence text must be found verbatim in source!"
                )

        # Confidence calculation
        conf_obj = None
        conf_band_str = "Low"
        q_score = None
        if dec.classification == "qualifies":
            # Logistic sigmoid centered at tau
            calibrated_p = 1.0 / (1.0 + math.exp(-7.0 * (dec.ranking_score - tau)))
            calibrated_p = max(0.01, min(0.99, calibrated_p))
            q_score = calibrated_p
            conf_band_str = "High" if calibrated_p >= 0.80 else ("Medium" if calibrated_p >= 0.50 else "Low")
            conf_obj = ConfidenceDetails(
                score=round(calibrated_p, 4),
                value=round(calibrated_p, 4),
                band=conf_band_str,
                method="platt_scaling",
                calibration_version="v3-logistic"
            )
        elif dec.classification == "does_not_qualify":
            # For confident negatives, emit uncalibrated or null unless fitted
            conf_obj = ConfidenceDetails(
                score=0.15,
                value=0.15,
                band="Low",
                method="heuristic_negative",
                calibration_version="v3-logistic"
            )
            q_score = 0.15
        else:
            # Insufficient evidence: NO pseudo-confidence on abstain!
            conf_obj = None
            conf_band_str = "Low"
            q_score = 0.10

        conditions_dict = {
            "entity": c1.model_dump() if hasattr(c1, "model_dump") else c1.dict(),
            "activity": c2.model_dump() if hasattr(c2, "model_dump") else c2.dict(),
            "system": c3.model_dump() if hasattr(c3, "model_dump") else c3.dict()
        }

        score_breakdown = ScoreBreakdown(
            positive_margin=c2.score,
            role_score=0.0 if not role_veto_reason else 0.95,
            final_score=dec.ranking_score,
            gate_flags=[dec.reason_code] if dec.reason_code else []
        )

        results.append({
            "api_version": "v3",
            "paper_id": paper_id,
            "title": title,
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "pub_year": pub_year,
            "classification": dec.classification,
            "classification_reason": dec.reason_code or dec.classification,
            "reason_code": dec.reason_code,
            "criterion_spec": spec.model_dump() if hasattr(spec, "model_dump") else (spec.dict() if hasattr(spec, "dict") else spec),
            "evidence": winning_ev_obj,
            "evidence_sentence": evidence_text,
            "evidence_status": dec.evidence_status,
            "evidence_source": "fulltext" if is_fulltext else "abstract_only",
            "evidence_offsets": evidence_offsets,
            "source_record_sha256": doc_sha,
            "text_sha256": doc_sha,
            "aliases_used": vetted_aliases,
            "retrieved_by": retrieved_by,
            "conditions": conditions_dict,
            "qualify_score": q_score,
            "confidence": conf_obj,
            "confidence_band": conf_band_str,
            "confidence_details": conf_obj,
            "confidence_kind": CONFIDENCE_KIND,
            "score_breakdown": score_breakdown,
            "guard_reason": dec.guard_reason,
            "dedup_of": dedup_of,
            "k_eligible": len(eligible_sentences),
            "k_scored": len(candidate_sentences),
            "reagent_signal": 0.0,
            "ranking_score": dec.ranking_score,
            "timings_ms": {}
        })

    timings["evaluation_ms"] = (time.perf_counter() - t_eval_start) * 1000.0

    # ==========================
    # Phase 7: Ranking
    # ==========================
    def v3_rank_key(item: Dict[str, Any]) -> Tuple[int, float, int, int, str]:
        cls = item["classification"]
        # Tier 0: Qualifiers
        # Tier 1: Scored non-qualifiers
        # Tier 2: Insufficient evidence (abstentions)
        # Tier 3: No text
        tier = 0 if cls == "qualifies" else (1 if cls == "does_not_qualify" else (2 if item.get("evidence_status") != "no_text" else 3))
        score = item.get("ranking_score", 0.0)
        # Count supported conditions
        conds = item.get("conditions", {})
        supp_count = sum(1 for c in conds.values() if isinstance(c, dict) and c.get("status") == "supported")
        pub_year_val = 0
        if item.get("pub_year") and str(item["pub_year"]).isdigit():
            pub_year_val = int(item["pub_year"])
        pid = str(item.get("paper_id") or "")
        return (tier, -score, -supp_count, -pub_year_val, pid)

    ranked_results = sorted(results, key=v3_rank_key)
    for rank_idx, r in enumerate(ranked_results, start=1):
        r["rank"] = rank_idx

    timings["total_execution_ms"] = (time.perf_counter() - t_start) * 1000.0

    return ClassificationRunResult(
        results=ranked_results,
        normalised_criterion=criterion.strip(),
        tau=tau,
        delta=delta,
        hypotheses_used=[spec.activity or "synthesis"],
        contrast_guard_active=True,
        guard_arming={"spec": spec.model_dump() if hasattr(spec, "model_dump") else spec.dict()},
        intent=intent,
        criterion_spec=spec,
        timings_ms=timings
    )
