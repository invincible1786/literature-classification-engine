"""Classification engine executing candidate extraction, intent gating, NLI inference, role guard, and sentence coupling."""
import hashlib
import logging
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from app.nli import NLIScorer
from app.text_utils import (
    canonicalize,
    split_sentences_with_spans,
    verify,
    build_entity_regex,
    vet_aliases,
    has_production_verb,
    is_review_article,
    detect_this_work_cues,
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
    .hypotheses_used, .contrast_guard_active, .guard_arming, .intent
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
        intent: str = "synthesis"
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
        return instance


def select_candidates_intent_aware(
    eligible_sentences: List[Sent],
    intent: str = "synthesis",
    synthesis_armed: Optional[bool] = None,
    max_k: int = DEFAULT_MAX_CANDIDATES
) -> List[Sent]:
    """
    Candidate selection is deterministic and intent-aware.
    - If synthesis: rank by production cues (action stems, yields, concentrations) then take top K.
    - If non-synthesis: position-balanced selection (first half + last half).
    - If eligible <= max_k: score all eligible sentences without pruning.
    Note: max_k serves as an operational latency cap, not a correctness filter.
    """
    if len(eligible_sentences) <= max_k:
        return eligible_sentences

    is_synth = synthesis_armed if synthesis_armed is not None else (intent == "synthesis")

    if is_synth:
        def synth_cue_score(s: Sent) -> Tuple[int, int]:
            text_lower = s.text.lower()
            stem_hits = sum(1 for stem in SYNTH_STEMS if stem in text_lower)
            quantity_hits = sum(1 for q in ["%", "g/l", "mg", "yield", "titer", "mol"] if q in text_lower)
            return (stem_hits + quantity_hits, -s.start)

        sorted_sents = sorted(eligible_sentences, key=synth_cue_score, reverse=True)
        selected = sorted_sents[:max_k]
        return sorted(selected, key=lambda s: s.start)
    else:
        k_front = (max_k + 1) // 2
        k_back = max_k // 2
        front_slice = eligible_sentences[:k_front]
        back_slice = eligible_sentences[-k_back:] if k_back > 0 else []
        selected_dict = {s.start: s for s in front_slice + back_slice}
        return sorted(selected_dict.values(), key=lambda s: s.start)


def classify_publications(
    publications: List[Dict[str, Any]],
    entity: str,
    criterion: str,
    aliases: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    contrast_delta: Optional[float] = None,
    custom_contrast_hypotheses: Optional[List[str]] = None,
    user_paraphrases: Optional[List[str]] = None,
    scorer: Optional[NLIScorer] = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    two_hit_rule: bool = False,
) -> ClassificationRunResult:
    """
    Classifies a list of Europe PMC publication records against the user criterion.
    Enforces:
    1. Sentence-level decision (evidence text strictly coupled to winning sentence).
    2. Intent-gated search and candidate verb requirements.
    3. Abstain path for reviews, lack of production verbs, and missing texts.
    4. Negative-role hypothesis and role pattern guard.
    5. Monotone Platt-calibrated confidence bands.
    """
    if scorer is None:
        raise ValueError("An NLIScorer instance must be provided to classify_publications.")

    tau = threshold if threshold is not None else DEFAULT_THRESHOLD
    delta = contrast_delta if contrast_delta is not None else DEFAULT_CONTRAST_DELTA

    intent = detect_intent(criterion, entity)
    synthesis_armed = (intent == "synthesis") or lexical_intent_armed(criterion, entity)

    guard_arming_meta = {
        "intent": intent,
        "lexical": synthesis_armed,
        "armed": synthesis_armed
    }

    # Build positive hypotheses ensemble and negative-role hypothesis
    hypotheses = build_hypotheses(
        criterion=criterion,
        entity=entity,
        synthesis_armed=synthesis_armed,
        user_paraphrases=user_paraphrases
    )
    negative_role_hyp = get_negative_role_hypothesis(entity)

    contrast_hypotheses = load_contrast_hypotheses(
        entity=entity,
        custom_hypotheses=custom_contrast_hypotheses
    )

    vetted_aliases = vet_aliases(aliases, entity)
    entity_regex = build_entity_regex(entity, vetted_aliases)

    # Candidate extraction per paper
    paper_meta_list: List[Dict[str, Any]] = []
    pairs_to_score: List[Tuple[str, str]] = []  # (premise, hypothesis)
    paper_pair_spans: List[Optional[Tuple[int, int]]] = []

    for pub in publications:
        raw_title = pub.get("title") or ""
        raw_abstract = pub.get("abstractText") or ""
        fulltext_secs = pub.get("fulltext_sections") or []

        canon_title = canonicalize(raw_title)
        canon_abstract = canonicalize(raw_abstract)

        full_canon = canon_abstract if canon_abstract else canon_title
        doc_sha = hashlib.sha256(full_canon.encode("utf-8")).hexdigest() if full_canon else None

        # Check if text is completely missing
        if not full_canon:
            paper_meta_list.append({
                "pub": pub,
                "status": "no_text",
                "canon_doc": "",
                "doc_sha": None,
                "scored_sents": [],
                "k_eligible": 0,
                "k_scored": 0
            })
            paper_pair_spans.append(None)
            continue

        # Review article pre-filter check
        if is_review_article(canon_title, canon_abstract):
            paper_meta_list.append({
                "pub": pub,
                "status": "review_abstain",
                "canon_doc": full_canon,
                "doc_sha": doc_sha,
                "scored_sents": [],
                "k_eligible": 0,
                "k_scored": 0
            })
            paper_pair_spans.append(None)
            continue

        # Segment sentences with exact span offsets
        sentences = split_sentences_with_spans(canon_abstract) if canon_abstract else []

        if fulltext_secs:
            offset_base = len(canon_abstract) + 1
            for p_text, sec_name in fulltext_secs:
                canon_p = canonicalize(p_text)
                if canon_p:
                    p_sents = split_sentences_with_spans(canon_p)
                    for s in p_sents:
                        sentences.append(
                            Sent(text=s.text, start=offset_base + s.start, end=offset_base + s.end)
                        )
                    offset_base += len(canon_p) + 1

        # Invariant: Sentence must contain target entity or vetted alias
        eligible_sents = [s for s in sentences if entity_regex.search(s.text)]

        # If abstract had none, try title
        if not eligible_sents and canon_title and entity_regex.search(canon_title):
            title_sents = split_sentences_with_spans(canon_title)
            eligible_sents = [s for s in title_sents if entity_regex.search(s.text)]
            full_canon = canon_title
            doc_sha = hashlib.sha256(full_canon.encode("utf-8")).hexdigest()

        k_eligible = len(eligible_sents)

        # Production verb check for synthesis intent
        if intent == "synthesis":
            eligible_sents = [s for s in eligible_sents if has_production_verb(s.text)]

        # If no qualifying candidate sentences survive
        if not eligible_sents:
            paper_meta_list.append({
                "pub": pub,
                "status": "insufficient",
                "canon_doc": full_canon,
                "doc_sha": doc_sha,
                "scored_sents": [],
                "k_eligible": k_eligible,
                "k_scored": 0
            })
            paper_pair_spans.append(None)
            continue

        # Intent-aware candidate selection (latency budgeting cap)
        scored_sents = select_candidates_intent_aware(
            eligible_sents,
            intent=intent,
            max_k=max_candidates
        )
        k_scored = len(scored_sents)

        meta = {
            "pub": pub,
            "status": "ok",
            "canon_doc": full_canon,
            "doc_sha": doc_sha,
            "scored_sents": scored_sents,
            "k_eligible": k_eligible,
            "k_scored": k_scored,
        }
        paper_meta_list.append(meta)

        # Register NLI pairs: for each candidate sentence, evaluate positive paraphrases + negative-role hypothesis
        start_idx = len(pairs_to_score)
        for s in scored_sents:
            for h in hypotheses:
                pairs_to_score.append((s.text, h))
            pairs_to_score.append((s.text, negative_role_hyp))
        end_idx = len(pairs_to_score)
        paper_pair_spans.append((start_idx, end_idx))

    # Batched inference
    all_nli_triples = scorer.score(pairs_to_score) if pairs_to_score else []

    # Decision logic per paper
    results: List[Dict[str, Any]] = []
    n_pos_hyps = len(hypotheses)
    stride = n_pos_hyps + 1  # positive paraphrases + negative-role hypothesis

    for idx, meta in enumerate(paper_meta_list):
        pub = meta["pub"]
        status = meta["status"]
        canon_doc = meta["canon_doc"]
        doc_sha = meta["doc_sha"]
        k_eligible = meta["k_eligible"]
        k_scored = meta["k_scored"]
        paper_id = str(pub.get("pmcid") or pub.get("pmid") or pub.get("doi") or "UNKNOWN")
        title = pub.get("title")
        doi = pub.get("doi")
        pmid = pub.get("pmid")
        pmcid = pub.get("pmcid")
        pub_year = pub.get("pubYear")
        dedup_of = pub.get("dedup_of")

        # Handle abstain / no-text cases
        if status == "no_text":
            results.append({
                "paper_id": paper_id,
                "title": title,
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "pub_year": pub_year,
                "classification": "does_not_qualify",
                "classification_reason": "no_usable_text",
                "evidence": None,
                "evidence_sentence": None,
                "evidence_status": "no_text",
                "evidence_source": "abstract",
                "evidence_offsets": None,
                "text_sha256": None,
                "qualify_score": None,
                "confidence": ConfidenceDetails(score=0.0, band="low"),
                "confidence_band": "low",
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
                "evidence_sentence": None,
                "evidence_status": "insufficient",
                "evidence_source": "abstract",
                "evidence_offsets": None,
                "text_sha256": doc_sha,
                "qualify_score": 0.15,
                "confidence": ConfidenceDetails(score=0.15, band="low"),
                "confidence_band": "low",
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
                "evidence_sentence": None,
                "evidence_status": "insufficient",
                "evidence_source": "abstract",
                "evidence_offsets": None,
                "text_sha256": doc_sha,
                "qualify_score": 0.12,
                "confidence": ConfidenceDetails(score=0.12, band="low"),
                "confidence_band": "low",
                "confidence_kind": CONFIDENCE_KIND,
                "score_breakdown": ScoreBreakdown(positive_margin=None, role_score=None, final_score=None, gate_flags=["no_production_verb"]),
                "guard_reason": None,
                "dedup_of": dedup_of,
                "k_eligible": k_eligible,
                "k_scored": 0,
                "reagent_signal": 0.0,
            })
            continue

        # Score candidate sentences for this paper
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

            # Take the best true paraphrase (per Section 6.8)
            pos_margin = max(cand_margins)
            p_ent = max(cand_p_ents)

            role_score = cand_role_triple[0]  # P(entail) on negative role

            gate_flags: List[str] = []
            # "This work" gate check
            has_pos_cue, has_bg_cue = detect_this_work_cues(sent_obj.text)
            margin_penalty = 0.0
            if has_bg_cue and not has_pos_cue:
                margin_penalty += 0.60
                gate_flags.append("this_work_background")
            elif has_pos_cue:
                gate_flags.append("this_work_positive")

            # Check role regex patterns
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

        # Sentence-Level Decision: Winner is argmax over candidate final_scores
        best_sent_eval = max(sentence_evaluations, key=lambda s: s.final_score)
        winning_sent = best_sent_eval.sent

        # STRICT COUPLING ASSERTION: Winning sentence must strictly match canonical slice
        assert verify(canon_doc, winning_sent), "Evidence text must match canonical text at character offsets!"
        assert winning_sent.text == canon_doc[winning_sent.start:winning_sent.end]

        # Runner-up sentence for ambiguity pooling
        sorted_evals = sorted(sentence_evaluations, key=lambda s: s.final_score, reverse=True)
        runner_up_final = sorted_evals[1].final_score if len(sorted_evals) > 1 else None

        # Affirmative contrast hypotheses check on the winning sentence
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

        # Two-hit rule check (optional ablation)
        single_borderline = False
        if two_hit_rule and (tau <= best_sent_eval.final_score < (tau + 0.10)):
            has_second_hit = any(
                idx != 0 and s.final_score >= (tau - 0.15)
                for idx, s in enumerate(sorted_evals)
            )
            if not has_second_hit:
                single_borderline = True

        # Decision
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

        # Verbatim evidence extraction
        evidence_source = "fulltext" if winning_sent.start >= len(canon_doc) else "abstract"
        evidence_obj = EvidenceSpan(
            text=winning_sent.text,
            char_start=winning_sent.start,
            char_end=winning_sent.end,
            source=evidence_source
        )

        # Calibrated Platt confidence calculation
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
            "confidence": ConfidenceDetails(score=conf_score or 0.0, band=conf_band),
            "confidence_band": conf_band,
            "confidence_kind": CONFIDENCE_KIND,
            "score_breakdown": score_breakdown,
            "guard_reason": guard_reason,
            "dedup_of": dedup_of,
            "k_eligible": k_eligible,
            "k_scored": k_scored,
            "reagent_signal": round(c_star, 4),
        })

    # Ranking
    ranked_results = sorted(results, key=ranking_sort_key)
    # Assign 1-based ranks
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
