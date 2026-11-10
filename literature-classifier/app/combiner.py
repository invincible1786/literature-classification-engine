"""Decision layer combiner executing strict 12-rule precedence (v3.2 WS17-R5) and ranking."""
from dataclasses import dataclass
from typing import Callable, List, Optional, Literal
from pydantic import BaseModel

from app.conditions import ConditionResult, EvidenceSpanV3
from app.criterion_parser import CriterionSpec
from app.reasons import validate_reason_code


class DecisionResult(BaseModel):
    classification: Literal["qualifies", "does_not_qualify", "insufficient_evidence"]
    reason_code: Optional[str] = None
    guard_reason: Optional[str] = None
    evidence_status: Literal["supported", "contradicted", "insufficient", "no_text"]
    primary_evidence: Optional[EvidenceSpanV3] = None
    ranking_score: float = 0.0


class CombinerInvariantError(RuntimeError):
    """Raised if the precedence rule table fails to terminate on an input state."""


@dataclass(frozen=True)
class CombinerInput:
    c1: ConditionResult
    c2: ConditionResult
    c3: ConditionResult
    spec: CriterionSpec
    has_usable_text: bool = True
    role_veto_reason: Optional[str] = None
    has_conflicting_synthesis_claim: bool = False
    is_pubtype_review_veto: bool = False
    is_evidence_split: bool = False


Rule = Callable[[CombinerInput], Optional[DecisionResult]]


def r1_no_text(inp: CombinerInput) -> Optional[DecisionResult]:
    """R1: Missing or empty abstract/fulltext."""
    if not inp.has_usable_text:
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="no_text",
            evidence_status="no_text",
            primary_evidence=None,
            ranking_score=-10.0,
        )
    return None


def r2_review_veto(inp: CombinerInput) -> Optional[DecisionResult]:
    """R2: Publication metadata review veto for this-study scope."""
    if inp.is_pubtype_review_veto and inp.spec.scope == "this_study":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code="review_article",
            guard_reason="pubtype_review_article_veto",
            evidence_status="contradicted",
            primary_evidence=None,
            ranking_score=-2.0,
        )
    return None


def r3a_reagent_veto(inp: CombinerInput) -> Optional[DecisionResult]:
    """R3a: Prepositional reagent/catalyst role veto (G1 only, no synthesis conflict)."""
    role_fired = inp.spec.entity_role == "product" and (
        inp.c1.reason_code == "reagent_role" or bool(inp.role_veto_reason)
    )
    if role_fired and not inp.has_conflicting_synthesis_claim:
        ev = inp.c1.evidence[0] if inp.c1.evidence else None
        return DecisionResult(
            classification="does_not_qualify",
            reason_code="reagent_role",
            guard_reason=inp.role_veto_reason or "reagent_role_prepositional_veto",
            evidence_status="contradicted",
            primary_evidence=ev,
            ranking_score=-2.0,
        )
    return None


def r3b_conflicting_evidence(inp: CombinerInput) -> Optional[DecisionResult]:
    """R3b: Conflicting role pattern + affirmative synthesis claim, or mixed-system conflict."""
    role_fired = inp.spec.entity_role == "product" and (
        inp.c1.reason_code == "reagent_role" or bool(inp.role_veto_reason)
    )
    if role_fired and inp.has_conflicting_synthesis_claim:
        ev = inp.c1.evidence[0] if inp.c1.evidence else (inp.c2.evidence[0] if inp.c2.evidence else None)
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="conflicting_evidence",
            guard_reason="reagent_role_with_affirmative_synthesis_claim",
            evidence_status="insufficient",
            primary_evidence=ev,
            ranking_score=-0.35,
        )
    if (
        inp.c1.reason_code == "conflicting_evidence"
        or inp.c2.reason_code == "conflicting_evidence"
        or inp.c3.reason_code == "conflicting_evidence"
    ):
        ev = (
            (inp.c2.evidence[0] if inp.c2.evidence else None)
            or (inp.c3.evidence[0] if inp.c3.evidence else None)
            or (inp.c1.evidence[0] if inp.c1.evidence else None)
        )
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="conflicting_evidence",
            guard_reason="condition_level_conflict",
            evidence_status="insufficient",
            primary_evidence=ev,
            ranking_score=-0.35,
        )
    return None


def r4_entity_absent(inp: CombinerInput) -> Optional[DecisionResult]:
    """R4: Entity confirmed absent in full text (contradicted) or unclear in abstract."""
    if inp.c1.status == "contradicted":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code=inp.c1.reason_code or "entity_absent",
            evidence_status="contradicted",
            primary_evidence=inp.c1.evidence[0] if inp.c1.evidence else None,
            ranking_score=-1.5,
        )
    if inp.c1.status == "unknown":
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code=inp.c1.reason_code or "entity_unclear",
            evidence_status="insufficient",
            primary_evidence=None,
            ranking_score=-0.5,
        )
    return None


def r5_background_citation(inp: CombinerInput) -> Optional[DecisionResult]:
    """R5: Prior-art attribution only (WS17-R2: evidence-based negative)."""
    if inp.c2.reason_code == "background_citation" and inp.c2.status != "supported":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code="background_citation",
            evidence_status="contradicted",
            primary_evidence=inp.c2.evidence[0] if inp.c2.evidence else None,
            ranking_score=-1.2,
        )
    return None


def r6_wrong_system(inp: CombinerInput) -> Optional[DecisionResult]:
    """R6: Required host system contradicted (full-text or abstract-explicit under D6)."""
    if inp.spec.system is not None and inp.c3.status == "contradicted":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code="wrong_system",
            evidence_status="contradicted",
            primary_evidence=inp.c3.evidence[0] if inp.c3.evidence else None,
            ranking_score=-1.0,
        )
    return None


def r7_system_unstated(inp: CombinerInput) -> Optional[DecisionResult]:
    """R7: Required host system not stated in abstract."""
    if inp.spec.system is not None and inp.c3.status == "unknown":
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="system_unstated",
            evidence_status="insufficient",
            primary_evidence=None,
            ranking_score=-0.2,
        )
    return None


def r8_different_activity(inp: CombinerInput) -> Optional[DecisionResult]:
    """R8: Activity contradicted (different_activity) or topical mention_only (WS17-R2)."""
    if inp.c2.status == "contradicted":
        code = inp.c2.reason_code or "different_activity"
        if not validate_reason_code(code):
            code = "different_activity"
        return DecisionResult(
            classification="does_not_qualify",
            reason_code=code,
            evidence_status="contradicted",
            primary_evidence=inp.c2.evidence[0] if inp.c2.evidence else None,
            ranking_score=-1.0,
        )
    if inp.c2.reason_code == "mention_only" and inp.c2.status != "supported":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code="mention_only",
            evidence_status="contradicted",
            primary_evidence=inp.c2.evidence[0] if inp.c2.evidence else None,
            ranking_score=-0.9,
        )
    return None


def r9_unstated_threshold(inp: CombinerInput) -> Optional[DecisionResult]:
    """R9: Quantitative threshold unstated, or remaining unclear/ambiguous activity abstains."""
    if inp.c2.reason_code == "unstated_threshold" and inp.c2.status != "supported":
        ev = inp.c2.evidence[0] if inp.c2.evidence else None
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="unstated_threshold",
            evidence_status="insufficient",
            primary_evidence=ev,
            ranking_score=-0.25,
        )
    if inp.c2.status == "unknown":
        code = inp.c2.reason_code or "activity_unclear"
        if code not in ("activity_unclear", "possible_review", "abstract_only", "evidence_split", "unstated_threshold"):
            code = "activity_unclear"
        ev = inp.c2.evidence[0] if inp.c2.evidence else None
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code=code,
            evidence_status="insufficient",
            primary_evidence=ev,
            ranking_score=-0.3,
        )
    return None


def r10_unsupported_criterion_clause(inp: CombinerInput) -> Optional[DecisionResult]:
    """R10: Allow-partial cap when unparsed criterion clauses remain after core checks."""
    if inp.spec.unparsed_clauses:
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="unsupported_criterion_clause",
            evidence_status="insufficient",
            primary_evidence=inp.c2.evidence[0] if inp.c2.evidence else None,
            ranking_score=-0.1,
        )
    return None


def r11_all_satisfied(inp: CombinerInput) -> Optional[DecisionResult]:
    """R11: All active conditions supported -> qualifies (unless evidence_split fallback flagged)."""
    if inp.is_evidence_split:
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="evidence_split",
            evidence_status="insufficient",
            primary_evidence=None,
            ranking_score=-0.1,
        )
    if (
        inp.c1.status == "supported"
        and inp.c2.status == "supported"
        and (inp.spec.system is None or inp.c3.status in ("supported", "not_required"))
    ):
        winning_ev = inp.c2.evidence[0] if inp.c2.evidence else (inp.c1.evidence[0] if inp.c1.evidence else None)
        req_scores = [inp.c1.score, inp.c2.score]
        if inp.spec.system is not None:
            req_scores.append(inp.c3.score)
        min_score = min(req_scores) if req_scores else inp.c2.score
        return DecisionResult(
            classification="qualifies",
            reason_code=None,
            evidence_status="supported",
            primary_evidence=winning_ev,
            ranking_score=min_score,
        )
    return None


RULES: List[Rule] = [
    r1_no_text,
    r2_review_veto,
    r3a_reagent_veto,
    r3b_conflicting_evidence,
    r4_entity_absent,
    r5_background_citation,
    r6_wrong_system,
    r7_system_unstated,
    r8_different_activity,
    r9_unstated_threshold,
    r10_unsupported_criterion_clause,
    r11_all_satisfied,
]


def combine(inp: CombinerInput) -> DecisionResult:
    """Execute the ordered 12-rule pure combiner table; totality invariant enforced."""
    for rule in RULES:
        decision = rule(inp)
        if decision is not None:
            return decision
    raise CombinerInvariantError(
        f"Combiner rules did not terminate for c1={inp.c1.status}, c2={inp.c2.status}, c3={inp.c3.status}"
    )


def combine_conditions(
    c1: ConditionResult,
    c2: ConditionResult,
    c3: ConditionResult,
    spec: CriterionSpec,
    has_usable_text: bool = True,
    role_veto_reason: Optional[str] = None,
    has_conflicting_synthesis_claim: bool = False,
    is_pubtype_review_veto: bool = False,
    is_evidence_split: bool = False,
) -> DecisionResult:
    """Compatibility wrapper around `combine(CombinerInput)`."""
    inp = CombinerInput(
        c1=c1,
        c2=c2,
        c3=c3,
        spec=spec,
        has_usable_text=has_usable_text,
        role_veto_reason=role_veto_reason,
        has_conflicting_synthesis_claim=has_conflicting_synthesis_claim,
        is_pubtype_review_veto=is_pubtype_review_veto,
        is_evidence_split=is_evidence_split,
    )
    return combine(inp)
