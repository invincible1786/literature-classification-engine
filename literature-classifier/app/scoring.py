"""Scoring, Platt scaling confidence calibration, monotone bands, and deterministic ranking."""
from typing import Optional, Tuple, Dict, Any, Union
import math

CONFIDENCE_KIND = "calibrated_platt_probability"
CALIBRATION_VERSION = "platt-v1-seed30"


def margin_to_qualify_score(m: Optional[float]) -> Optional[float]:
    """
    Monotonically map NLI margin m in [-1.0, 1.0] to qualify_score in [0.0, 1.0].
    Returns None if m is None (e.g. insufficient text or retrieval failure).
    """
    if m is None or math.isnan(m):
        return None
    score = (m + 1.0) / 2.0
    return round(max(0.0, min(1.0, score)), 4)


def score_to_calibrated_prob(final_score: Optional[float], tau: float = 0.45) -> float:
    """
    Platt scaling mapping final_score in [-1.0, 1.0] to calibrated P(correct) in [0.0, 1.0].
    Sigmoid scaling centered at tau:
    P(correct) = 1 / (1 + exp(-7.0 * (final_score - tau)))
    """
    if final_score is None:
        return 0.0
    logit = 7.0 * (final_score - tau)
    logit = max(-15.0, min(15.0, logit))
    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(float(prob), 4)


def guard_confidence(
    c_star: float,
    p_ent: float,
    delta: float,
    cap: float = 0.65
) -> float:
    """
    Monotonic confidence formula for affirmative contrast guard rejections.
    T = p_ent - delta
    raw = (c_star - T) / (1.0 - T)
    confidence = clamp(raw, 0.0, cap)
    """
    T = p_ent - delta
    denom = max(1e-9, 1.0 - T)
    raw = (c_star - T) / denom
    return round(max(0.0, min(cap, raw)), 4)


def calculate_confidence(
    m_star: Optional[float],
    tau: float,
    decision: str,
    guard_triggered: bool = False,
    c_star: float = 0.0,
    p_ent: float = 0.0,
    delta: float = 0.00
) -> Optional[float]:
    """
    Calculate calibrated probability confidence score.
    Returns None if m_star is None.
    """
    if m_star is None or math.isnan(m_star):
        return None

    if guard_triggered:
        return guard_confidence(c_star=c_star, p_ent=p_ent, delta=delta, cap=0.65)

    if decision == "qualifies":
        return score_to_calibrated_prob(m_star, tau=tau)
    else:
        # Distance from threshold below tau
        raw_prob = score_to_calibrated_prob(m_star, tau=tau)
        # For non-qualifiers, higher confidence corresponds to being well below threshold
        inv_prob = 1.0 - raw_prob
        return round(max(0.0, min(1.0, inv_prob)), 4)


def assign_confidence_band(
    confidence: Optional[float],
    best_margin: Optional[float] = None,
    runner_up_margin: Optional[float] = None
) -> str:
    """
    Assign qualitative confidence band derived purely and monotonically from probability:
    - 'high':   >= 0.80
    - 'medium': 0.50 <= P < 0.80
    - 'low':    < 0.50
    Applies runner-up ambiguity pooling: if top-2 sentence margin difference is < 0.10,
    a 'high' band is dampened to 'medium'.
    """
    if confidence is None:
        return "low"

    if confidence >= 0.80:
        band = "high"
    elif confidence >= 0.50:
        band = "medium"
    else:
        band = "low"

    # Ambiguity pooling penalty
    if (
        band == "high"
        and best_margin is not None
        and runner_up_margin is not None
        and (best_margin - runner_up_margin) < 0.10
    ):
        band = "medium"

    return band


def ranking_sort_key(item: Union[Dict[str, Any], Any]) -> Tuple:
    """
    Deliberate deterministic sorting key:
    1. Qualifiers first (qualifies < does_not_qualify)
    2. Papers with no usable text (no_text) strictly last
    3. Scored non-qualifiers before guard-rejected / abstained non-qualifiers
    4. Calibrated confidence descending
    5. qualify_score descending
    6. pub_year descending
    7. paper_id ascending (deterministic tie-breaker)
    """
    if isinstance(item, dict):
        is_qual = (item.get("classification") == "qualifies")
        status = str(item.get("evidence_status") or "")
        reason = str(item.get("classification_reason") or "")
        guard_reason = item.get("guard_reason")
        q_score = item.get("qualify_score")
        conf_obj = item.get("confidence")
        if isinstance(conf_obj, dict):
            conf_val = conf_obj.get("score", 0.0)
        elif isinstance(conf_obj, (int, float)):
            conf_val = float(conf_obj)
        else:
            conf_val = 0.0
        year_str = item.get("pub_year")
        paper_id = str(item.get("paper_id") or "")
    else:
        is_qual = (getattr(item, "classification", "") == "qualifies")
        status = str(getattr(item, "evidence_status", "") or "")
        reason = str(getattr(item, "classification_reason", "") or "")
        guard_reason = getattr(item, "guard_reason", None)
        q_score = getattr(item, "qualify_score", None)
        conf_obj = getattr(item, "confidence", None)
        if hasattr(conf_obj, "score"):
            conf_val = conf_obj.score
        elif isinstance(conf_obj, (int, float)):
            conf_val = float(conf_obj)
        else:
            conf_val = 0.0
        year_str = getattr(item, "pub_year", None)
        paper_id = str(getattr(item, "paper_id", "") or "")

    guard_rejected = bool(
        "contrast_guard" in reason
        or "role_pattern" in reason
        or guard_reason is not None
    )
    is_no_text = (status == "no_text" or (q_score is None and "insufficient_text" in reason))
    is_abstain = (status == "insufficient")

    score_num = float(q_score) if (q_score is not None) else 0.0
    conf_num = float(conf_val) if (conf_val is not None) else 0.0

    try:
        year_val = int(year_str or 0)
    except (ValueError, TypeError):
        year_val = 0

    return (
        0 if is_qual else 1,                   # 1. Qualifiers first
        1 if is_no_text else 0,                # 2. No-text strictly last
        1 if (is_abstain or guard_rejected) else 0, # 3. Scored non-qualifiers before guard/abstain
        -conf_num,                             # 4. Confidence descending
        -score_num,                            # 5. qualify_score descending
        -year_val,                             # 6. pub_year descending
        paper_id                               # 7. tie-breaker paper_id ascending
    )


# Backward-compatibility aliases for legacy callers
calculate_qualify_score = margin_to_qualify_score
get_confidence_band = assign_confidence_band


def rank_papers(results: list) -> list:
    return sorted(results, key=ranking_sort_key)
