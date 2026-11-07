"""Condition checkers for v3 architecture: C1 Entity, C2 Activity, C3 System."""
import re
import logging
from typing import List, Dict, Any, Optional, Literal, Tuple
from pydantic import BaseModel, Field

from app.schemas import EvidenceSpan
from app.criterion_parser import CriterionSpec, SystemSpec, SYSTEMS_DB
from app.text_utils import (
    Sent,
    build_entity_regex,
    vet_aliases,
    detect_this_work_cues,
    clean_markup_and_unescape,
    map_canonical_span_to_raw,
    has_production_verb,
)
from app.hypotheses import check_role_patterns

logger = logging.getLogger(__name__)


class EvidenceSpanV3(BaseModel):
    role: Literal["entity", "activity", "system"]
    raw_text: str
    display_text: str
    raw_start: int
    raw_end: int
    sentence_index: int
    source: str = "abstract"
    verifiable_alone: bool = True

    # Backwards compatibility properties
    @property
    def text(self) -> str:
        return self.display_text

    @property
    def char_start(self) -> int:
        return self.raw_start

    @property
    def char_end(self) -> int:
        return self.raw_end

    def to_v2_span(self) -> EvidenceSpan:
        return EvidenceSpan(
            text=self.display_text,
            char_start=self.raw_start,
            char_end=self.raw_end,
            source=self.source
        )


class ConditionResult(BaseModel):
    name: Literal["entity", "activity", "system"]
    status: Literal["supported", "contradicted", "unknown", "not_required"]
    score: float = 0.0  # raw model/feature score, NOT a probability
    evidence: List[EvidenceSpanV3] = Field(default_factory=list)
    also_mentions: List[str] = Field(default_factory=list)
    reason_code: Optional[str] = None


def make_evidence_span(
    sent: Sent,
    role: Literal["entity", "activity", "system"],
    raw_text: str,
    verifiable_alone: bool = True
) -> EvidenceSpanV3:
    raw_start, raw_end = map_canonical_span_to_raw(raw_text, sent.start, sent.end)
    extracted_raw = raw_text[raw_start:raw_end] if raw_text else sent.text
    display_text = clean_markup_and_unescape(extracted_raw) or sent.text
    return EvidenceSpanV3(
        role=role,
        raw_text=extracted_raw,
        display_text=display_text,
        raw_start=raw_start,
        raw_end=raw_end,
        sentence_index=sent.sentence_index,
        source=sent.source,
        verifiable_alone=verifiable_alone
    )


# Related compound guard: reject prefixes like dihydro-, demethyl-, hydroxy-
_RELATED_PREFIXES = [
    r"dihydro", r"demethyl", r"hydroxy", r"deoxy", r"nor", r"homo", r"epi",
    r"iso", r"pseudo", r"allo", r"neo", r"anhydro"
]


def is_related_compound_collision(sentence: str, entity: str) -> bool:
    """Detect if match in sentence is actually a modified related chemical derivative."""
    clean_ent = entity.strip().lower()
    for prefix in _RELATED_PREFIXES:
        pattern = rf"\b{prefix}[- ]?{re.escape(clean_ent)}\b"
        if re.search(pattern, sentence, re.IGNORECASE):
            # Check if standalone unmodified entity also occurs
            standalone = rf"(?<!\w){re.escape(clean_ent)}(?!\w)"
            matches = list(re.finditer(standalone, sentence, re.IGNORECASE))
            # If every occurrence is preceded by the prefix, it's a collision
            if all(m.start() >= len(prefix) and sentence[m.start()-len(prefix):m.start()].lower().startswith(prefix) for m in matches):
                return True
    return False


def check_c1_entity(
    sentences: List[Sent],
    raw_text: str,
    spec: CriterionSpec,
    aliases: Optional[List[str]] = None,
    is_fulltext: bool = False
) -> Tuple[ConditionResult, Optional[str]]:
    """
    Evaluate C1 Entity condition.
    Returns: (ConditionResult, optional_role_veto_reason)
    """
    vetted_aliases = vet_aliases(aliases, spec.entity)
    entity_regex = build_entity_regex(spec.entity, vetted_aliases)

    matched_sents: List[Sent] = []
    role_pattern_matched = None
    role_pattern_span = None

    for s in sentences:
        if entity_regex.search(s.text):
            if is_related_compound_collision(s.text, spec.entity):
                continue
            matched_sents.append(s)

            # Check role patterns
            pattern_hit = check_role_patterns(s.text, entity_regex)
            if pattern_hit:
                role_pattern_matched = pattern_hit
                role_pattern_span = s

    # Role Veto handling:
    # Veto applies only when entity_role == "product" and entity is object of role preposition
    if spec.entity_role == "product" and role_pattern_matched and role_pattern_span:
        span_obj = make_evidence_span(role_pattern_span, "entity", raw_text, verifiable_alone=True)
        return (
            ConditionResult(
                name="entity",
                status="contradicted",
                score=-0.95,
                evidence=[span_obj],
                reason_code="reagent_role"
            ),
            f"role_pattern: {role_pattern_matched}"
        )

    # Agent role handling:
    # If entity_role == "agent", matching role patterns is SUPPORTING evidence!
    if spec.entity_role == "agent":
        if matched_sents:
            ev_list = [make_evidence_span(s, "entity", raw_text, verifiable_alone=True) for s in matched_sents[:2]]
            score_val = 0.90 if role_pattern_matched else 0.70
            return (
                ConditionResult(
                    name="entity",
                    status="supported",
                    score=score_val,
                    evidence=ev_list,
                    reason_code=None
                ),
                None
            )

    if matched_sents:
        ev_list = [make_evidence_span(s, "entity", raw_text, verifiable_alone=True) for s in matched_sents[:2]]
        return (
            ConditionResult(
                name="entity",
                status="supported",
                score=1.0,
                evidence=ev_list,
                reason_code=None
            ),
            None
        )

    # Entity was not matched in provided text
    if is_fulltext:
        return (
            ConditionResult(
                name="entity",
                status="contradicted",
                score=-1.0,
                evidence=[],
                reason_code="entity_absent"
            ),
            None
        )
    else:
        # Abstract-only: entity may be named only in Methods/Results
        return (
            ConditionResult(
                name="entity",
                status="unknown",
                score=0.0,
                evidence=[],
                reason_code="entity_unclear"
            ),
            None
        )


def check_c2_activity(
    candidate_sentences: List[Sent],
    raw_text: str,
    spec: CriterionSpec,
    nli_scorer: Any,
    tau_activity: float = 0.45,
    tau_review: float = 0.60
) -> ConditionResult:
    """
    Evaluate C2 Activity condition using NLI competing hypotheses,
    experimental attribution cues (p_this_work), and review soft-scoring.
    """
    if not candidate_sentences:
        return ConditionResult(
            name="activity",
            status="unknown",
            score=0.0,
            evidence=[],
            reason_code="mention_only"
        )

    # Build primary hypothesis
    if spec.entity_role == "agent":
        kind_str = spec.agent_kind or "catalyst"
        act_clause = f" in {spec.activity}" if spec.activity else ""
        h_primary = f"{spec.entity} is used as a {kind_str}{act_clause} in this study."
        h_alternatives = [
            f"{spec.entity} is the product of the reaction.",
            f"{spec.entity} is only mentioned or analyzed."
        ]
    else:
        act_term = spec.activity or "synthesis"
        h_primary = f"{spec.entity} was synthesized, produced, or isolated in this study."
        h_alternatives = [
            f"{spec.entity} was used as a reagent, catalyst, or starting material.",
            f"{spec.entity} was purchased or obtained commercially.",
            f"{spec.entity} was evaluated for clinical or pharmacological effects."
        ]

    best_score = -999.0
    best_sent = None
    best_reason = None
    contradicting_sent = None

    for sent in candidate_sentences:
        # Score primary hypothesis
        probs = nli_scorer.predict_proba([(sent.text, h_primary)])[0]
        p_ent = probs.get("entailment", 0.0)
        p_con = probs.get("contradiction", 0.0)
        p_neu = probs.get("neutral", 0.0)
        margin = p_ent - p_con

        # Score alternatives
        alt_probs = [nli_scorer.predict_proba([(sent.text, alt)])[0].get("entailment", 0.0) for alt in h_alternatives]
        max_alt = max(alt_probs) if alt_probs else 0.0

        # Attribution / "This Work" cues
        has_pos, has_bg = detect_this_work_cues(sent.text)
        
        # Soft review score check
        review_words = ["we review", "this review", "systematic review", "synthesize current knowledge"]
        review_score = 0.8 if any(rw in sent.text.lower() for rw in review_words) else 0.0

        # Attribution adjustments
        p_this_work = 0.5
        if has_pos:
            p_this_work += 0.4
        if has_bg:
            p_this_work -= 0.5

        final_sent_score = margin - max_alt

        # Apply demotions if scope is this_study
        sent_reason = None
        if spec.scope == "this_study":
            if has_bg and not has_pos:
                final_sent_score = min(final_sent_score, tau_activity - 0.10)
                sent_reason = "background_citation"
            elif review_score >= tau_review and p_this_work < 0.6:
                final_sent_score = min(final_sent_score, tau_activity - 0.10)
                sent_reason = "possible_review"

        if max_alt > 0.85 and p_ent < 0.20 and has_pos:
            contradicting_sent = sent

        if final_sent_score > best_score:
            best_score = final_sent_score
            best_sent = sent
            best_reason = sent_reason

    # Decide status
    if best_score >= tau_activity and best_sent:
        ev_span = make_evidence_span(best_sent, "activity", raw_text, verifiable_alone=True)
        return ConditionResult(
            name="activity",
            status="supported",
            score=best_score,
            evidence=[ev_span],
            reason_code=None
        )

    if contradicting_sent and best_score < 0.10:
        ev_span = make_evidence_span(contradicting_sent, "activity", raw_text, verifiable_alone=True)
        return ConditionResult(
            name="activity",
            status="contradicted",
            score=best_score,
            evidence=[ev_span],
            reason_code="different_activity"
        )

    # Unknown
    fallback_reason = best_reason or ("background_citation" if best_sent and detect_this_work_cues(best_sent.text)[1] else "activity_unclear")
    ev_list = [make_evidence_span(best_sent, "activity", raw_text, verifiable_alone=True)] if best_sent else []
    return ConditionResult(
        name="activity",
        status="unknown",
        score=max(0.0, best_score),
        evidence=ev_list,
        reason_code=fallback_reason
    )


def check_c3_system(
    sentences: List[Sent],
    raw_text: str,
    spec: CriterionSpec,
    is_fulltext: bool = False,
    abstract_negative_policy: Literal["conservative", "strict"] = "conservative"
) -> ConditionResult:
    """
    Evaluate C3 System condition using gazetteer and mention verification.
    """
    if spec.system is None:
        return ConditionResult(
            name="system",
            status="not_required",
            score=1.0,
            evidence=[],
            also_mentions=[],
            reason_code=None
        )

    req_name = spec.system.name.lower()
    req_family = spec.system.family.lower() if spec.system.family else None
    req_synonyms = [s.lower() for s in spec.system.synonyms] + [req_name]
    if req_family:
        req_synonyms.extend([m.lower() for m in SYSTEMS_DB.get("families", {}).get(req_family, [])])
    req_synonyms = list(set(req_synonyms))

    supported_sents: List[Sent] = []
    other_organisms: List[str] = []
    other_org_sent: Optional[Sent] = None

    # Scan for mentions across sentences
    all_known_organisms = []
    for fam, mems in SYSTEMS_DB.get("families", {}).items():
        all_known_organisms.extend(mems)

    for s in sentences:
        s_lower = s.text.lower()
        # Check requested system match
        if any(re.search(rf"\b{re.escape(syn)}\b", s_lower) for syn in req_synonyms):
            supported_sents.append(s)

        # Check other organisms
        for org in all_known_organisms:
            if org not in req_synonyms:
                if re.search(rf"\b{re.escape(org)}\b", s_lower):
                    if org not in other_organisms:
                        other_organisms.append(org)
                        if not other_org_sent:
                            other_org_sent = s

    if supported_sents:
        ev_list = [make_evidence_span(s, "system", raw_text, verifiable_alone=False) for s in supported_sents[:2]]
        return ConditionResult(
            name="system",
            status="supported",
            score=1.0,
            evidence=ev_list,
            also_mentions=other_organisms,
            reason_code=None
        )

    # Requested system not found
    if other_organisms and other_org_sent:
        ev_list = [make_evidence_span(other_org_sent, "system", raw_text, verifiable_alone=False)]
        if is_fulltext or abstract_negative_policy == "strict":
            return ConditionResult(
                name="system",
                status="contradicted",
                score=-0.90,
                evidence=ev_list,
                also_mentions=other_organisms,
                reason_code="wrong_system"
            )
        else:
            # Abstract-only with conservative policy
            return ConditionResult(
                name="system",
                status="unknown",
                score=0.10,
                evidence=ev_list,
                also_mentions=other_organisms,
                reason_code="system_mismatch_abstract"
            )

    return ConditionResult(
        name="system",
        status="unknown",
        score=0.0,
        evidence=[],
        also_mentions=[],
        reason_code="system_unstated"
    )
