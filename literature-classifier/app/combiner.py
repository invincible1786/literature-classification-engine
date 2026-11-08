"""Decision layer combiner executing strict 11-step precedence rules and ranking."""
from typing import Dict, Any, Optional, Tuple, Literal
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


def combine_conditions(
    c1: ConditionResult,
    c2: ConditionResult,
    c3: ConditionResult,
    spec: CriterionSpec,
    has_usable_text: bool = True,
    role_veto_reason: Optional[str] = None,
    is_pubtype_review_veto: bool = False,
    is_evidence_split: bool = False
) -> DecisionResult:
    """
    Execute strict 11-rule precedence combiner (Section 6.1):
    
    1  No usable text -> insufficient_evidence / no_text
    2  Role veto fired -> does_not_qualify / reagent_role
    3  pubType review veto fired -> does_not_qualify / review_article
    4a Entity contradicted -> does_not_qualify / entity_absent
    4b Entity unknown -> insufficient_evidence / entity_unclear
    5  Activity contradicted -> does_not_qualify / different_activity
    6  System required, contradicted -> does_not_qualify / wrong_system
    6b System required, contradicted, abstract-only -> insufficient_evidence / system_mismatch_abstract
    7  Activity unknown -> insufficient_evidence / reason_code
    8  System required, unknown -> insufficient_evidence / system_unstated
    9  Unsupported clause present -> insufficient_evidence / unsupported_criterion_clause
    10 Activity span lacks entity -> insufficient_evidence / evidence_split
    11 Entity and activity supported; system supported or not_required -> qualifies / none
    """
    # 1. No usable text
    if not has_usable_text:
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="no_text",
            evidence_status="no_text",
            primary_evidence=None,
            ranking_score=-10.0
        )

    # 2. Role veto fired (applies only when entity_role == "product")
    if spec.entity_role == "product" and (c1.reason_code == "reagent_role" or role_veto_reason):
        ev = c1.evidence[0] if c1.evidence else None
        return DecisionResult(
            classification="does_not_qualify",
            reason_code="reagent_role",
            guard_reason=role_veto_reason or "reagent_role_prepositional_veto",
            evidence_status="contradicted",
            primary_evidence=ev,
            ranking_score=-2.0
        )

    # 3. pubType review veto fired
    if is_pubtype_review_veto and spec.scope == "this_study":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code="review_article",
            guard_reason="pubtype_review_article_veto",
            evidence_status="contradicted",
            primary_evidence=None,
            ranking_score=-2.0
        )

    # 4a. Entity contradicted (absent from full text)
    if c1.status == "contradicted":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code=c1.reason_code or "entity_absent",
            evidence_status="contradicted",
            primary_evidence=c1.evidence[0] if c1.evidence else None,
            ranking_score=-1.5
        )

    # 4b. Entity unknown
    if c1.status == "unknown":
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code=c1.reason_code or "entity_unclear",
            evidence_status="insufficient",
            primary_evidence=None,
            ranking_score=-0.5
        )

    # 5. Activity contradicted (evidence-based negative)
    # Note: only fires if activity is NOT supported!
    if c2.status == "contradicted":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code=c2.reason_code or "different_activity",
            evidence_status="contradicted",
            primary_evidence=c2.evidence[0] if c2.evidence else None,
            ranking_score=-1.0
        )

    # 6. System required, contradicted
    if spec.system is not None and c3.status == "contradicted":
        return DecisionResult(
            classification="does_not_qualify",
            reason_code=c3.reason_code or "wrong_system",
            evidence_status="contradicted",
            primary_evidence=c3.evidence[0] if c3.evidence else None,
            ranking_score=-1.0
        )

    # 6b. System required, contradicted, abstract-only (conservative policy)
    if spec.system is not None and c3.reason_code == "system_mismatch_abstract":
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="system_mismatch_abstract",
            evidence_status="insufficient",
            primary_evidence=c3.evidence[0] if c3.evidence else None,
            ranking_score=-0.4
        )

    # 7. Activity unknown
    if c2.status == "unknown":
        ev = c2.evidence[0] if c2.evidence else None
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code=c2.reason_code or "activity_unclear",
            evidence_status="insufficient",
            primary_evidence=ev,
            ranking_score=-0.3
        )

    # 8. System required, unknown
    if spec.system is not None and c3.status == "unknown":
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code=c3.reason_code or "system_unstated",
            evidence_status="insufficient",
            primary_evidence=None,
            ranking_score=-0.2
        )

    # 9. Unsupported clause present (allow_partial_criterion)
    if spec.unparsed_clauses:
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="unsupported_criterion_clause",
            evidence_status="insufficient",
            primary_evidence=c2.evidence[0] if c2.evidence else None,
            ranking_score=-0.1
        )

    # 10. Activity span lacks entity (evidence only via "the product")
    if is_evidence_split:
        return DecisionResult(
            classification="insufficient_evidence",
            reason_code="evidence_split",
            evidence_status="insufficient",
            primary_evidence=None,
            ranking_score=-0.1
        )

    # 11. Entity and activity supported; system supported or not_required -> qualifies
    if c1.status == "supported" and c2.status == "supported" and c3.status in ("supported", "not_required"):
        # Winning evidence span is activity span
        winning_ev = c2.evidence[0] if c2.evidence else (c1.evidence[0] if c1.evidence else None)
        # Ranking score = min over supported required conditions
        req_scores = [c1.score, c2.score]
        if spec.system is not None:
            req_scores.append(c3.score)
        min_score = min(req_scores) if req_scores else c2.score

        return DecisionResult(
            classification="qualifies",
            reason_code=None,
            evidence_status="supported",
            primary_evidence=winning_ev,
            ranking_score=min_score
        )

    # Catch-all fallback
    return DecisionResult(
        classification="insufficient_evidence",
        reason_code="activity_unclear",
        evidence_status="insufficient",
        primary_evidence=None,
        ranking_score=0.0
    )

