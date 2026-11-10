"""Condition checkers for v3.2 architecture: C1 Entity, C2 Activity, C3 System."""
import re
import logging
from typing import List, Dict, Any, Optional, Literal, Tuple
from pydantic import BaseModel, Field

from app.schemas import EvidenceSpan
from app.criterion_parser import (
    CriterionSpec,
    SystemSpec,
    load_systems_gazetteer,
    normalize_numeric_value,
)
from app.text_utils import (
    Sent,
    CLAUSE_SPLIT_MIN,
    COMPOUND_PREFIXES,
    MIN_EVIDENCE_CHARS,
    FINITE_PRODUCTION_VERBS,
    build_entity_regex,
    vet_aliases,
    detect_this_work_cues,
    clean_markup_and_unescape,
    map_canonical_span_to_raw,
    has_production_verb,
    has_finite_production_verb,
    is_related_compound_collision,
    split_clauses_bracket_aware,
    resolve_anaphora,
)
from app.hypotheses import check_role_patterns

logger = logging.getLogger(__name__)


class EvidenceSpanV3(BaseModel):
    role: Literal["entity", "activity", "system"]
    raw_text: str
    display_text: str
    raw_start: int
    raw_end: int
    canon_start: Optional[int] = None
    canon_end: Optional[int] = None
    sentence_index: int
    source: str = "abstract"
    verifiable_alone: bool = True

    @property
    def text(self) -> str:
        return self.display_text

    @property
    def char_start(self) -> int:
        return self.canon_start if self.canon_start is not None else self.raw_start

    @property
    def char_end(self) -> int:
        return self.canon_end if self.canon_end is not None else self.raw_end

    def to_v2_span(self) -> EvidenceSpan:
        return EvidenceSpan(
            text=self.display_text,
            char_start=self.char_start,
            char_end=self.char_end,
            raw_start=self.raw_start,
            raw_end=self.raw_end,
            raw_text=self.raw_text,
            display_text=self.display_text,
            sentence_index=self.sentence_index,
            source=self.source,
            role=self.role,
            verifiable_alone=self.verifiable_alone,
        )


class ConditionResult(BaseModel):
    name: Literal["entity", "activity", "system"]
    status: Literal["supported", "contradicted", "unknown", "not_required"]
    score: float = 0.0  # raw model/feature score, NOT a probability
    neg_margin: float = 0.0  # negative margin for D5 negative calibration
    evidence: List[EvidenceSpanV3] = Field(default_factory=list)
    also_mentions: List[str] = Field(default_factory=list)
    reason_code: Optional[str] = None
    attached_clauses: List[int] = Field(default_factory=list)
    comparative_mentions: List[int] = Field(default_factory=list)
    synthesis_claim_sents: List[int] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


def make_evidence_span(
    sent: Sent,
    role: Literal["entity", "activity", "system"],
    raw_text: str,
    verifiable_alone: bool = True
) -> EvidenceSpanV3:
    # F3: Build raw_text as exact slice of canonicalized body
    # Collapse whitespace before slicing so canon_body[start:end] == display_text == clean(raw_text)
    display_text = sent.text
    return EvidenceSpanV3(
        role=role,
        raw_text=sent.text,
        display_text=display_text,
        raw_start=sent.start,
        raw_end=sent.end,
        canon_start=sent.start,
        canon_end=sent.end,
        sentence_index=sent.sentence_index,
        source=sent.source,
        verifiable_alone=verifiable_alone
    )


# F5: Unified chemical compound prefix list (registered in config/constants.md)
_RELATED_PREFIXES = COMPOUND_PREFIXES


def check_f8_structural_guards(sentence_text: str, entity: str, title: Optional[str] = None) -> Optional[str]:
    """
    Check bypass/negation and degradation stems on parent sentence + title (F8).
    Returns reason string if matched, else None.
    """
    text_to_check = f"{title or ''} {sentence_text or ''}".lower()
    clean_ent = entity.strip().lower()
    e_pat = re.escape(clean_ent)

    # 1. Bypass/negation: without (going through|using|passing through) {E}, bypass*, not via
    bypass_patterns = [
        rf"\bwithout\s+(?:going\s+through|using|passing\s+through)\s+(?:[\w\-]+\s+){{0,2}}{e_pat}\b",
        rf"\bwithout\s+(?:[\w\-]+\s+){{0,2}}{e_pat}\b",
        rf"\bbypass\w*\s+(?:[\w\-]+\s+){{0,2}}{e_pat}\b",
        rf"\bnot\s+via\s+(?:[\w\-]+\s+){{0,2}}{e_pat}\b",
    ]
    for p in bypass_patterns:
        m = re.search(p, text_to_check)
        if m:
            return f"bypass_route: {m.group(0)}"

    # 2. Degradation stems: degrad*, decaffein*, catabol*
    deg_stems = [r"\bdegrad\w*", r"\bdecaffein\w*", r"\bcatabol\w*"]
    for stem in deg_stems:
        deg_match = (
            re.search(rf"{stem}\s+(?:[\w\-]+\s+){{0,2}}{e_pat}\b", text_to_check)
            or re.search(rf"\b{e_pat}\s+(?:[\w\-]+\s+){{0,2}}{stem}\b", text_to_check)
            or re.search(r"\bdecaffein\w*\b", text_to_check)
        )
        if deg_match:
            return f"degradation_guard: {deg_match.group(0)}"
        if re.search(stem, text_to_check) and re.search(rf"\b{e_pat}\b", text_to_check):
            if re.search(rf"{stem}.{{1,50}}\b{e_pat}\b|\b{e_pat}\b.{{1,50}}{stem}", text_to_check):
                return f"degradation_guard: {stem}"

    return None


def check_c1_entity(
    sentences: List[Sent],
    raw_text: str,
    spec: CriterionSpec,
    aliases: Optional[List[str]] = None,
    is_fulltext: bool = False
) -> Tuple[ConditionResult, Optional[str]]:
    """
    Evaluate C1 Entity condition with WS14-E anaphora fallback.
    Returns: (ConditionResult, optional_role_veto_reason)
    """
    vetted_aliases = vet_aliases(aliases, spec.entity)
    entity_regex = build_entity_regex(spec.entity, vetted_aliases)

    matched_sents: List[Sent] = []
    role_pattern_matched = None
    role_pattern_span = None
    non_veto_sents: List[Sent] = []

    for s in sentences:
        if entity_regex.search(s.text):
            if is_related_compound_collision(s.text, spec.entity):
                continue
            matched_sents.append(s)

            pattern_hit = check_role_patterns(s.text, entity_regex)
            if pattern_hit:
                role_pattern_matched = pattern_hit
                role_pattern_span = s
            else:
                non_veto_sents.append(s)

    # Anaphora resolution for follow-up sentences (WS14-E)
    anaphora_map = resolve_anaphora(sentences, entity_regex)
    if anaphora_map and matched_sents:
        for idx, _ant_idx in anaphora_map.items():
            s = sentences[idx]
            if s not in matched_sents:
                matched_sents.append(s)
                non_veto_sents.append(s)

    diag = {
        "anaphora_resolved": anaphora_map,
        "non_veto_sent_indices": [s.sentence_index for s in non_veto_sents],
    }

    # Role Veto handling (G1 product only; downstream R3a/R3b resolves conflicts via C2)
    if spec.entity_role == "product" and role_pattern_matched and role_pattern_span:
        span_obj = make_evidence_span(role_pattern_span, "entity", raw_text, verifiable_alone=True)
        return (
            ConditionResult(
                name="entity",
                status="contradicted",
                score=-0.95,
                evidence=[span_obj],
                reason_code="reagent_role",
                diagnostics=diag,
            ),
            role_pattern_matched,
        )

    # Agent role handling (G2 agent)
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
                    reason_code=None,
                    diagnostics=diag,
                ),
                None,
            )

    if matched_sents:
        ev_list = [make_evidence_span(s, "entity", raw_text, verifiable_alone=True) for s in matched_sents[:2]]
        return (
            ConditionResult(
                name="entity",
                status="supported",
                score=1.0,
                evidence=ev_list,
                reason_code=None,
                diagnostics=diag,
            ),
            None,
        )

    if is_fulltext:
        return (
            ConditionResult(
                name="entity",
                status="contradicted",
                score=-1.0,
                evidence=[],
                reason_code="entity_absent",
                diagnostics=diag,
            ),
            None,
        )

    return (
        ConditionResult(
            name="entity",
            status="unknown",
            score=0.0,
            evidence=[],
            reason_code="entity_unclear",
            diagnostics=diag,
        ),
        None,
    )


_RE_OBS_QUANTITY = re.compile(
    r"\b(?:yield|titer|titre|accumulat(?:ed|ing)|produc(?:ed|ing)|conversion|purity|concentration)\s*(?:of|was|reached|up\s+to|at|=|:)?\s*"
    r"(\d+(?:\.\d+)?)\s*(g/L|mg/L|ug/L|μg/L|mol%|%)\b|"
    r"(\d+(?:\.\d+)?)\s*(g/L|mg/L|ug/L|μg/L|mol%|%)\s+(?:of\s+[a-zA-Z0-9\-\s]+|yield|titer|conversion)",
    re.IGNORECASE,
)


def _check_numeric_thresholds(
    sentences: List[Sent],
    spec: CriterionSpec,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Verify G3 numeric_threshold constraints against all available sentences (WS14-D).
    Returns None if satisfied, ('unstated_threshold', None) if missing, or
    ('different_activity', detail_str) if explicitly violated.
    """
    num_constraints = [c for c in spec.constraints if c.kind == "numeric_threshold"]
    if not num_constraints:
        return None

    full_text = " ".join(s.text for s in sentences)
    observed: List[Tuple[float, str]] = []
    for m in _RE_OBS_QUANTITY.finditer(full_text):
        val_str = m.group(1) or m.group(3)
        unit_str = m.group(2) or m.group(4)
        if val_str and unit_str:
            n_val, n_unit, ok = normalize_numeric_value(float(val_str), unit_str)
            if ok:
                observed.append((n_val, n_unit))

    if not observed:
        return ("unstated_threshold", None)

    for c in num_constraints:
        if c.value is None or not c.unit:
            continue
        matching_unit_vals = [v for v, u in observed if u == c.unit]
        if not matching_unit_vals:
            return ("unstated_threshold", None)
        passed = False
        for v in matching_unit_vals:
            if c.operator == ">" and v > c.value:
                passed = True
            elif c.operator == ">=" and v >= c.value:
                passed = True
            elif c.operator == "<" and v < c.value:
                passed = True
            elif c.operator == "<=" and v <= c.value:
                passed = True
        if not passed:
            return ("different_activity", f"observed {matching_unit_vals} {c.unit} fails {c.operator} {c.value} {c.unit}")

    return None


def _check_exclusions(clause_text: str, spec: CriterionSpec) -> Optional[str]:
    """Check whether supported clause matches an excluded method constraint (WS14-D)."""
    exclusions = [c for c in spec.constraints if c.kind == "exclusion"]
    if not exclusions:
        return None
    low = clause_text.lower()
    for exc in exclusions:
        target = exc.target.lower().strip()
        if not target:
            continue
        if target in low:
            return exc.target
        # Handle "chemical synthesis" excluded -> chemically synthesized / total synthesis
        if "chemical" in target and re.search(r"\bchemical(?:ly)?\s+synthes", low):
            return exc.target
    return None


def check_c2_activity(
    candidate_sentences: List[Sent],
    raw_text: str,
    spec: CriterionSpec,
    nli_scorer: Any,
    tau_activity: float = 0.45,
    tau_review: float = 0.60,
    all_sentences: Optional[List[Sent]] = None,
    anaphora_map: Optional[Dict[int, int]] = None,
    entity_regex: Optional[re.Pattern] = None,
    title: Optional[str] = None,
) -> ConditionResult:
    """
    Evaluate C2 Activity condition (v3.3):
    - Bracket-aware clause splitting replacing 250-char truncation (WS15-B)
    - Clause-local experimental attribution p_this_work (WS15-C)
    - Anaphora antecedent premise pairing (WS14-E)
    - Conflicting synthesis claim detection for R3b routing (WS14-A)
    - G3 exclusion & numeric threshold checking (WS14-D)
    - F4: Clauses are auxiliary scoring units; parent Punkt sentence is always public evidence span
    - F5: Entity-anchor gate (anaphora-safe; rejects COMPOUND_PREFIXES collisions)
    - F6: MIN_EVIDENCE_CHARS >= 30 and finite action verb check
    - F7: Contrast hypotheses for degradation and measurement
    - F8: Minimal structural guards checked on parent sentence + title
    """
    if not candidate_sentences:
        return ConditionResult(
            name="activity",
            status="unknown",
            score=0.0,
            neg_margin=0.0,
            evidence=[],
            reason_code="mention_only",
        )

    if spec.entity_role == "agent":
        kind_str = spec.agent_kind or "catalyst"
        act_clause = f" in {spec.activity}" if spec.activity else ""
        h_primary = f"{spec.entity} is used as a {kind_str}{act_clause} in this study."
        h_alternatives = [
            f"{spec.entity} is the product of the reaction.",
            f"{spec.entity} is only mentioned or analyzed.",
            f"{spec.entity} was degraded, metabolized, or broken down in this study.",
            f"{spec.entity} content was measured or quantified in this study.",
        ]
    else:
        h_primary = f"{spec.entity} was synthesized, produced, or isolated in this study."
        h_alternatives = [
            f"{spec.entity} was used as a reagent, catalyst, or starting material.",
            f"{spec.entity} was purchased or obtained commercially.",
            f"{spec.entity} was evaluated for clinical or pharmacological effects.",
            f"{spec.entity} was degraded, metabolized, or broken down in this study.",
            f"{spec.entity} content was measured or quantified in this study.",
        ]

    # Lookup map for anaphoric antecedents
    sent_by_idx: Dict[int, Sent] = {s.sentence_index: s for s in (all_sentences or candidate_sentences)}
    if entity_regex is None:
        entity_regex = build_entity_regex(spec.entity)

    # Expand candidate sentences via bracket-depth clause splitting (WS15-B)
    clause_items: List[Tuple[Sent, Sent, str, bool]] = []  # (cl, parent_sent, premise_text_for_NLI, parent_has_role)
    for sent in candidate_sentences:
        clauses = split_clauses_bracket_aware(sent, min_len=CLAUSE_SPLIT_MIN)
        ant_idx = anaphora_map.get(sent.sentence_index) if anaphora_map else None
        ant_sent = sent_by_idx.get(ant_idx) if ant_idx is not None else None
        parent_has_role = bool(check_role_patterns(sent.text, entity_regex))
        for cl in clauses:
            # F4 gate: A clause is scored ONLY if it:
            # (a) contains the entity or an alias, or has a resolved antecedent in anaphora_map, and
            # (b) contains a finite production verb. Clauses failing this are skipped for scoring, never returned.
            has_ent = bool(entity_regex.search(cl.text)) or (sent.sentence_index in anaphora_map if anaphora_map else False)
            has_verb = has_finite_production_verb(cl.text)
            if len(clauses) > 1:
                if not (has_ent and has_verb):
                    continue
            else:
                if not has_verb:
                    continue

            premise = f"{ant_sent.text} {cl.text}" if ant_sent is not None else cl.text
            clause_items.append((cl, sent, premise, parent_has_role))

    if not clause_items:
        return ConditionResult(
            name="activity",
            status="unknown",
            score=0.0,
            neg_margin=0.0,
            evidence=[],
            reason_code="activity_unclear",
            diagnostics={"clause_gate": "no_clause_with_entity_and_finite_verb"}
        )

    # Batch NLI scoring across all clauses
    primary_pairs = [(premise, h_primary) for _, _, premise, _ in clause_items]
    primary_probs = nli_scorer.predict_proba(primary_pairs)

    alt_probs_by_item: List[List[float]] = [[] for _ in clause_items]
    for alt_h in h_alternatives:
        alt_res = nli_scorer.predict_proba([(premise, alt_h) for _, _, premise, _ in clause_items])
        for i, pr in enumerate(alt_res):
            alt_probs_by_item[i].append(pr.get("entailment", 0.0))

    best_score = -999.0
    best_neg_margin = 0.0
    best_clause: Optional[Sent] = None
    best_parent: Optional[Sent] = None
    best_reason: Optional[str] = None
    contradicting_clause: Optional[Sent] = None
    contradicting_parent: Optional[Sent] = None
    contradicting_neg_margin = 0.0
    synthesis_claim_sents: List[int] = []
    bg_only_clause: Optional[Sent] = None

    for idx, (cl, parent_sent, _premise, parent_has_role) in enumerate(clause_items):
        probs = primary_probs[idx]
        p_ent = probs.get("entailment", 0.0)
        p_con = probs.get("contradiction", 0.0)
        margin = p_ent - p_con

        item_alts = alt_probs_by_item[idx] if alt_probs_by_item[idx] else []
        max_alt = max(item_alts) if item_alts else 0.0
        neg_m = max_alt + p_con - p_ent

        # Role alternatives (reagent, purchased, clinical) penalize synthesis margin
        role_alts = item_alts[:3] if len(item_alts) >= 3 else item_alts
        max_role_alt = max(role_alts) if role_alts else 0.0

        # Clause-local attribution (WS15-C)
        has_pos, has_bg = detect_this_work_cues(cl.text)
        review_words = ["we review", "this review", "systematic review", "synthesize current knowledge"]
        review_score = 0.8 if any(rw in cl.text.lower() for rw in review_words) else 0.0

        p_this_work = 0.5
        if has_pos:
            p_this_work += 0.4
        if has_bg:
            p_this_work -= 0.5

        final_cl_score = margin - max_role_alt
        cl_reason = None

        if spec.scope == "this_study":
            if has_bg and not has_pos:
                final_cl_score = min(final_cl_score, tau_activity - 0.10)
                cl_reason = "background_citation"
                if margin >= 0.30 and bg_only_clause is None:
                    bg_only_clause = cl
            elif review_score >= tau_review and p_this_work < 0.5:
                final_cl_score = min(final_cl_score, tau_activity - 0.10)
                cl_reason = "possible_review"

        # Conflicting synthesis claim detection (WS14-A): clean affirmative sentence without role pattern
        if final_cl_score >= tau_activity and not parent_has_role and not check_role_patterns(cl.text, entity_regex):
            if cl.sentence_index not in synthesis_claim_sents:
                synthesis_claim_sents.append(cl.sentence_index)

        # F7: Alternative hypothesis outscores target hypothesis on evidence span -> different_activity
        alt_outscores_primary = False
        if len(item_alts) > 3:
            p_degrad = item_alts[3]
            p_meas = item_alts[4] if len(item_alts) > 4 else 0.0
            if (p_degrad > p_ent and p_degrad >= 0.60) or (p_degrad >= 0.70 and p_ent < 0.50):
                alt_outscores_primary = True
            elif (p_meas > p_ent and p_meas >= 0.70 and p_ent < tau_activity):
                alt_outscores_primary = True

        if (((max_alt >= 0.70 and p_ent < 0.20) or alt_outscores_primary or (p_con >= 0.75 and (p_con - p_ent) >= 0.65)) and not (has_bg and not has_pos)):
            if contradicting_clause is None or max(neg_m, max_alt, p_con - p_ent) > contradicting_neg_margin:
                contradicting_clause = cl
                contradicting_parent = parent_sent
                contradicting_neg_margin = max(neg_m, max_alt, p_con - p_ent)

        if final_cl_score > best_score:
            best_score = final_cl_score
            best_neg_margin = neg_m
            best_clause = cl
            best_parent = parent_sent
            best_reason = cl_reason

    # 1. Qualified activity candidate
    if best_score >= tau_activity and best_parent:
        ev_span = make_evidence_span(best_parent, "activity", raw_text, verifiable_alone=True)

        # F6: Minimum evidence quality: returned evidence must be >= MIN_EVIDENCE_CHARS (30)
        # and contain a finite action verb
        if len(best_parent.text.strip()) < MIN_EVIDENCE_CHARS or not has_finite_production_verb(best_parent.text):
            return ConditionResult(
                name="activity",
                status="unknown",
                score=best_score,
                neg_margin=0.0,
                evidence=[ev_span],
                reason_code="activity_unclear",
                synthesis_claim_sents=synthesis_claim_sents,
                diagnostics={"quality_fail": "below_min_chars_or_no_finite_verb"}
            )

        # F5: Entity-anchor gate (anaphora-safe)
        ant_idx = anaphora_map.get(best_parent.sentence_index) if anaphora_map else None
        ant_sent = sent_by_idx.get(ant_idx) if ant_idx is not None else None

        parent_has_entity = bool(entity_regex.search(best_parent.text))
        parent_has_collision = is_related_compound_collision(best_parent.text, spec.entity)

        ant_has_entity = bool(entity_regex.search(ant_sent.text)) if ant_sent else False
        ant_has_collision = is_related_compound_collision(ant_sent.text, spec.entity) if ant_sent else False

        if parent_has_collision or (not parent_has_entity and ant_has_collision):
            return ConditionResult(
                name="activity",
                status="unknown",
                score=best_score,
                neg_margin=0.0,
                evidence=[ev_span],
                reason_code="entity_unclear",
                synthesis_claim_sents=[],
                diagnostics={"collision": "compound_prefix_collision"}
            )

        if not parent_has_entity and not ant_has_entity:
            return ConditionResult(
                name="activity",
                status="unknown",
                score=best_score,
                neg_margin=0.0,
                evidence=[ev_span],
                reason_code="entity_unclear",
                synthesis_claim_sents=[],
                diagnostics={"anchor_fail": "entity_not_found"}
            )

        diag = {}
        if not parent_has_entity and ant_has_entity and ant_sent:
            diag["resolved_antecedent"] = {
                "sentence_index": ant_sent.sentence_index,
                "text": ant_sent.text,
                "start": ant_sent.start,
                "end": ant_sent.end
            }

        # F8: Minimal structural guards checked on parent sentence + title
        f8_hit = check_f8_structural_guards(best_parent.text, spec.entity, title=title)
        if f8_hit:
            return ConditionResult(
                name="activity",
                status="contradicted",
                score=-0.90,
                neg_margin=0.90,
                evidence=[ev_span],
                reason_code="different_activity",
                synthesis_claim_sents=[],
                diagnostics={"structural_guard": f8_hit, "resolved_antecedent": diag.get("resolved_antecedent")}
            )

        # F11: Route exclusions checked on parent sentence + title
        excl_hit = _check_exclusions(f"{title or ''} {best_parent.text}", spec)
        if excl_hit:
            return ConditionResult(
                name="activity",
                status="contradicted",
                score=-0.85,
                neg_margin=0.85,
                evidence=[ev_span],
                reason_code="different_activity",
                synthesis_claim_sents=[],
                diagnostics={"excluded_method": excl_hit, "resolved_antecedent": diag.get("resolved_antecedent")}
            )

        # Check G3 numeric thresholds (WS14-D)
        num_check = _check_numeric_thresholds(all_sentences or candidate_sentences, spec)
        if num_check is not None:
            num_reason, num_detail = num_check
            if num_reason == "unstated_threshold":
                return ConditionResult(
                    name="activity",
                    status="unknown",
                    score=best_score,
                    neg_margin=0.0,
                    evidence=[ev_span],
                    reason_code="unstated_threshold",
                    synthesis_claim_sents=synthesis_claim_sents,
                    diagnostics={"threshold_status": "unstated", "resolved_antecedent": diag.get("resolved_antecedent")},
                )
            else:
                return ConditionResult(
                    name="activity",
                    status="contradicted",
                    score=-0.80,
                    neg_margin=0.80,
                    evidence=[ev_span],
                    reason_code="different_activity",
                    synthesis_claim_sents=synthesis_claim_sents,
                    diagnostics={"threshold_failed": num_detail, "resolved_antecedent": diag.get("resolved_antecedent")},
                )

        return ConditionResult(
            name="activity",
            status="supported",
            score=best_score,
            neg_margin=best_neg_margin,
            evidence=[ev_span],
            reason_code=None,
            synthesis_claim_sents=synthesis_claim_sents,
            diagnostics=diag,
        )

    # 2. Explicit contradicting activity vs categorical mention_only (WS17-R4)
    if contradicting_parent and best_score < 0.10:
        ev_span = make_evidence_span(contradicting_parent, "activity", raw_text, verifiable_alone=True)
        is_survey_mention = bool(
            not has_production_verb(contradicting_parent.text)
            and re.search(r"\b(?:questionnaire|recorded|intake|survey|consumption|dietary|prevalence|cohort)\b", contradicting_parent.text, re.I)
        )
        has_exp_neg_verb = has_production_verb(contradicting_parent.text) or bool(
            re.search(
                r"\b(?:no|not|failed|without|evaluated|treated|administered|assayed|catalyzed|inhibited|degraded|incorporated|quantified|tested)\b",
                contradicting_parent.text,
                re.IGNORECASE,
            )
        )
        if is_survey_mention or (not has_exp_neg_verb and contradicting_neg_margin < 0.70):
            return ConditionResult(
                name="activity",
                status="unknown",
                score=0.0,
                neg_margin=0.0,
                evidence=[ev_span],
                reason_code="mention_only",
                synthesis_claim_sents=synthesis_claim_sents,
            )
        return ConditionResult(
            name="activity",
            status="contradicted",
            score=best_score,
            neg_margin=contradicting_neg_margin,
            evidence=[ev_span],
            reason_code="different_activity",
            synthesis_claim_sents=synthesis_claim_sents,
        )

    # 3. Prior-art / background citation demotion (WS15-C + WS17-R2)
    if best_reason == "background_citation" or bg_only_clause is not None:
        chosen_cl = bg_only_clause or best_clause
        chosen_parent = sent_by_idx.get(chosen_cl.sentence_index) if chosen_cl else None
        parent_for_bg = chosen_parent or chosen_cl
        ev_list = [make_evidence_span(parent_for_bg, "activity", raw_text, verifiable_alone=True)] if parent_for_bg else []
        return ConditionResult(
            name="activity",
            status="contradicted",
            score=best_score,
            neg_margin=best_neg_margin,
            evidence=ev_list,
            reason_code="background_citation",
            synthesis_claim_sents=synthesis_claim_sents,
        )

    # 4. Ambiguous / topical mention fallback
    fallback_parent = best_parent or (sent_by_idx.get(best_clause.sentence_index) if best_clause else None)
    if best_clause is not None and not best_reason and best_score < 0.15 and not has_production_verb(best_clause.text):
        is_survey_or_mention = bool(
            re.search(r"\b(?:questionnaire|recorded|intake|survey|consumption|dietary|prevalence|cohort)\b", best_clause.text, re.I)
        )
        if is_survey_or_mention:
            ev_list = [make_evidence_span(fallback_parent, "activity", raw_text, verifiable_alone=True)] if fallback_parent else []
            return ConditionResult(
                name="activity",
                status="unknown",
                score=0.0,
                neg_margin=0.0,
                evidence=ev_list,
                reason_code="mention_only",
                synthesis_claim_sents=synthesis_claim_sents,
            )
    fallback_reason = best_reason or "activity_unclear"
    ev_list = [make_evidence_span(fallback_parent, "activity", raw_text, verifiable_alone=True)] if fallback_parent else []
    return ConditionResult(
        name="activity",
        status="unknown",
        score=max(0.0, best_score),
        neg_margin=best_neg_margin,
        evidence=ev_list,
        reason_code=fallback_reason,
        synthesis_claim_sents=synthesis_claim_sents,
    )


_RE_COMPARATIVE_CLAUSE = re.compile(
    r"^\s*(?:unlike|in\s+contrast\s+(?:to|with)|compared\s+(?:to|with)|whereas|rather\s+than|instead\s+of)\b",
    re.IGNORECASE,
)


def _match_family_for_token(org: str, db: Dict[str, Any]) -> Optional[str]:
    low = org.lower()
    for fam, mems in db.get("families", {}).items():
        if low in [m.lower() for m in mems]:
            return fam
    return None


def check_c3_system(
    sentences: List[Sent],
    raw_text: str,
    spec: CriterionSpec,
    is_fulltext: bool = False,
    abstract_negative_policy: Literal["strict", "conservative"] = "strict",
    entity_regex: Optional[re.Pattern] = None,
) -> ConditionResult:
    """
    Evaluate C3 System condition (v3.2):
    - Clause-attached organism matching (WS14-B)
    - Comparative/concessive exclusion ('unlike E. coli, ...' does not satisfy E. coli)
    - Distinct in_vitro / cell-free family separation from microbial
    - D6 policy: explicit different host in abstract -> wrong_system (contradicted)
    """
    if spec.system is None:
        return ConditionResult(
            name="system",
            status="not_required",
            score=1.0,
            evidence=[],
            also_mentions=[],
            reason_code=None,
        )

    db = load_systems_gazetteer()
    req_name = spec.system.name.lower()
    req_family = spec.system.family.lower() if spec.system.family else None
    req_synonyms = [s.lower() for s in spec.system.synonyms] + [req_name]
    if req_family:
        req_synonyms.extend([m.lower() for m in db.get("families", {}).get(req_family, [])])
        if req_family == "microbial":
            for sub_fam in ("yeast", "bacteria", "fungi"):
                req_synonyms.extend([m.lower() for m in db.get("families", {}).get(sub_fam, [])])
    req_synonyms = list(set(req_synonyms))

    all_known_organisms: List[str] = []
    for _fam, mems in db.get("families", {}).items():
        for m in mems:
            if m.lower() not in all_known_organisms:
                all_known_organisms.append(m.lower())

    supported_clauses: List[Sent] = []
    other_organisms: List[str] = []
    comparative_orgs: List[str] = []
    other_org_clause: Optional[Sent] = None
    attached_clauses: List[int] = []
    comparative_mentions: List[int] = []
    saw_in_vitro = False
    saw_cellular = False

    for s in sentences:
        # Split sentence at bracket-depth-0 boundaries to inspect clause-level attachment (WS14-B)
        clauses = split_clauses_bracket_aware(s, min_len=0)
        parent_has_context = (
            (entity_regex is not None and bool(entity_regex.search(s.text)))
            or has_production_verb(s.text)
            or len(clauses) == 1
        )
        for cl in clauses:
            cl_low = cl.text.lower()
            if _RE_COMPARATIVE_CLAUSE.search(cl.text):
                # Comparative/concessive clause: record organism mentions as context only, never C3 support or wrong_system
                for org in all_known_organisms:
                    if re.search(rf"\b{re.escape(org)}\b", cl_low):
                        if cl.sentence_index not in comparative_mentions:
                            comparative_mentions.append(cl.sentence_index)
                        if org not in req_synonyms and org not in comparative_orgs:
                            comparative_orgs.append(org)
                continue

            # Check clause attachment to activity/entity
            clause_attached = (
                (entity_regex is not None and bool(entity_regex.search(cl.text)))
                or has_production_verb(cl.text)
                or parent_has_context
            )

            # Check requested system match
            req_hit = any(re.search(rf"\b{re.escape(syn)}\b", cl_low) for syn in req_synonyms)
            if req_hit and clause_attached:
                supported_clauses.append(cl)
                if cl.sentence_index not in attached_clauses:
                    attached_clauses.append(cl.sentence_index)

            # Track other / alternative organisms and in_vitro vs cellular usage
            for org in all_known_organisms:
                if re.search(rf"\b{re.escape(org)}\b", cl_low):
                    fam = _match_family_for_token(org, db)
                    if fam == "in_vitro":
                        saw_in_vitro = True
                    elif fam in ("microbial", "yeast", "bacteria", "fungi", "plant", "mammalian"):
                        saw_cellular = True

                    if org not in req_synonyms and clause_attached:
                        if org not in other_organisms:
                            other_organisms.append(org)
                        if other_org_clause is None:
                            other_org_clause = cl

    all_also_mentions = list(dict.fromkeys(other_organisms + comparative_orgs))

    # Mixed in_vitro vs cellular claim check (WS14-B)
    if supported_clauses and req_family == "in_vitro" and saw_cellular and not any(
        _match_family_for_token(o, db) == "in_vitro" for o in other_organisms
    ):
        if any(_match_family_for_token(o, db) in ("microbial", "yeast", "bacteria") for o in other_organisms):
            p_s = next((s for s in sentences if s.sentence_index == supported_clauses[0].sentence_index), supported_clauses[0])
            ev_list = [make_evidence_span(p_s, "system", raw_text, verifiable_alone=False)]
            return ConditionResult(
                name="system",
                status="unknown",
                score=0.2,
                evidence=ev_list,
                also_mentions=all_also_mentions,
                reason_code="conflicting_evidence",
                attached_clauses=attached_clauses,
                comparative_mentions=comparative_mentions,
            )

    if supported_clauses:
        # F4: parent sentence is always the public evidence span
        ev_list = []
        for c in supported_clauses[:2]:
            p_s = next((s for s in sentences if s.sentence_index == c.sentence_index), c)
            ev_list.append(make_evidence_span(p_s, "system", raw_text, verifiable_alone=False))
        return ConditionResult(
            name="system",
            status="supported",
            score=1.0,
            evidence=ev_list,
            also_mentions=all_also_mentions,
            reason_code=None,
            attached_clauses=attached_clauses,
            comparative_mentions=comparative_mentions,
        )

    # Requested system not found; explicit alternative organism stated in non-comparative clause
    if other_organisms and other_org_clause:
        p_s = next((s for s in sentences if s.sentence_index == other_org_clause.sentence_index), other_org_clause)
        ev_list = [make_evidence_span(p_s, "system", raw_text, verifiable_alone=False)]
        if is_fulltext or abstract_negative_policy == "strict":
            return ConditionResult(
                name="system",
                status="contradicted",
                score=-0.90,
                neg_margin=0.90,
                evidence=ev_list,
                also_mentions=all_also_mentions,
                reason_code="wrong_system",
                attached_clauses=attached_clauses,
                comparative_mentions=comparative_mentions,
            )
        return ConditionResult(
            name="system",
            status="unknown",
            score=0.10,
            evidence=ev_list,
            also_mentions=all_also_mentions,
            reason_code="system_unstated",
            attached_clauses=attached_clauses,
            comparative_mentions=comparative_mentions,
        )

    return ConditionResult(
        name="system",
        status="unknown",
        score=0.0,
        evidence=[],
        also_mentions=all_also_mentions,
        reason_code="system_unstated",
        attached_clauses=attached_clauses,
        comparative_mentions=comparative_mentions,
    )
