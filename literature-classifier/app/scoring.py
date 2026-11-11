"""Scoring, MLE-fitted Platt scaling confidence calibration, monotone bands, and deterministic ranking."""
import hashlib
import json
import math
import os
from typing import Optional, Tuple, Dict, Any, Union

CONFIDENCE_KIND = "calibrated_platt_probability"
CALIBRATION_VERSION = "platt-v3.3-mle"

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_PATH = os.path.join(_BASE_DIR, "config", "calibration.json")
SPLITS_PATH = os.path.join(_BASE_DIR, "eval", "splits.json")

_calibration_cache: Optional[Dict[str, Any]] = None


class CalibrationConfigError(RuntimeError):
    """Raised when config/calibration.json is missing, contains legacy placeholders, or fails splits audit."""


def _compute_splits_sha256() -> Optional[str]:
    if not os.path.exists(SPLITS_PATH):
        return None
    h = hashlib.sha256()
    with open(SPLITS_PATH, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def is_placeholder_params(a: float, b: float) -> bool:
    """Detect legacy hardcoded placeholder constants (-7.0, 3.15) or (7.0, -3.15)."""
    return (
        (abs(a - (-7.0)) < 1e-6 and abs(b - 3.15) < 1e-6)
        or (abs(a - 7.0) < 1e-6 and abs(b - (-3.15)) < 1e-6)
    )


def load_calibration_config(strict_splits: bool = False, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load MLE-fitted Platt scaling parameters from config/calibration.json (WS12-A / WS17-R3).
    Fails loudly if the file is absent, uses legacy hardcoded placeholders, or fails splits_sha256 audit.
    """
    global _calibration_cache
    if _calibration_cache is not None and not force_reload and not strict_splits:
        return _calibration_cache

    if not os.path.exists(CALIBRATION_PATH):
        raise CalibrationConfigError(
            f"Missing calibration config at {CALIBRATION_PATH}. Run `python -m eval.calibrate` first."
        )

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    qual = cfg.get("qualifier_path", cfg)
    a_val = qual.get("A")
    b_val = qual.get("B")
    if a_val is None or b_val is None:
        raise CalibrationConfigError("calibration.json missing qualifier_path parameters (A, B).")
    if is_placeholder_params(float(a_val), float(b_val)):
        raise CalibrationConfigError(
            f"Legacy placeholder calibration parameters detected (A={a_val}, B={b_val}). "
            "Run `python -m eval.calibrate` to fit MLE parameters from dev splits."
        )

    if strict_splits:
        recorded_sha = cfg.get("splits_sha256")
        actual_sha = _compute_splits_sha256()
        if not recorded_sha or not actual_sha or recorded_sha != actual_sha:
            raise CalibrationConfigError(
                f"splits_sha256 mismatch: calibration.json has {recorded_sha}, eval/splits.json has {actual_sha}."
            )

    _calibration_cache = cfg
    return cfg


def reload_calibration_config() -> Dict[str, Any]:
    return load_calibration_config(force_reload=True)


def _sigmoid_platt(margin: float, a: float, b: float) -> float:
    """Evaluate P = 1 / (1 + exp(-(A * m + B))) monotonically clamped to [0.01, 0.99]."""
    logit = a * margin + b
    logit = max(-15.0, min(15.0, logit))
    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(max(0.01, min(0.99, float(prob))), 4)


def calibrate_qualifier(margin: Optional[float], strict_splits: bool = False) -> Optional[float]:
    """Compute calibrated P(correct) on the qualifier path using MLE-fitted (A, B)."""
    if margin is None or math.isnan(margin):
        return None
    cfg = load_calibration_config(strict_splits=strict_splits)
    qual = cfg.get("qualifier_path", cfg)
    return _sigmoid_platt(float(margin), float(qual["A"]), float(qual["B"]))


def calibrate_negative(neg_margin: Optional[float], strict_splits: bool = False) -> Optional[float]:
    """
    Compute calibrated P(correct) on the evidence-based negative path
    (different_activity, wrong_system only — WS17-R4 / D5).
    Returns None if negative_path is not fitted.
    """
    if neg_margin is None or math.isnan(neg_margin):
        return None
    cfg = load_calibration_config(strict_splits=strict_splits)
    neg = cfg.get("negative_path")
    if not neg or not neg.get("fitted", False):
        return None
    return _sigmoid_platt(float(neg_margin), float(neg["A"]), float(neg["B"]))


def margin_to_qualify_score(m: Optional[float]) -> Optional[float]:
    """Monotonically map NLI margin m in [-1.0, 1.0] to qualify_score in [0.0, 1.0]."""
    if m is None or math.isnan(m):
        return None
    score = (m + 1.0) / 2.0
    return round(max(0.0, min(1.0, score)), 4)


def score_to_calibrated_prob(final_score: Optional[float], tau: float = 0.45) -> float:
    """Backward-compatible wrapper computing calibrated probability via MLE Platt map."""
    if final_score is None:
        return 0.0
    p = calibrate_qualifier(final_score)
    return p if p is not None else 0.0


def guard_confidence(
    c_star: float,
    p_ent: float,
    delta: float,
    cap: float = 0.65
) -> float:
    """Monotonic confidence formula for v2 affirmative contrast guard rejections."""
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
    """Legacy v2 confidence helper."""
    if m_star is None or math.isnan(m_star):
        return None
    if guard_triggered:
        return guard_confidence(c_star=c_star, p_ent=p_ent, delta=delta, cap=0.65)
    if decision == "qualifies":
        return score_to_calibrated_prob(m_star, tau=tau)
    raw_prob = score_to_calibrated_prob(m_star, tau=tau)
    return round(max(0.0, min(1.0, 1.0 - raw_prob)), 4)


def assign_confidence_band(
    confidence: Optional[float],
    best_margin: Optional[float] = None,
    runner_up_margin: Optional[float] = None
) -> str:
    """
    Assign qualitative confidence band derived monotonically from probability:
    - 'high':   >= 0.80
    - 'medium': 0.50 <= P < 0.80
    - 'low':    < 0.50
    Applies runner-up ambiguity pooling (< 0.10 margin diff dampens high -> medium).
    """
    if confidence is None:
        return "low"

    if confidence >= 0.80:
        band = "high"
    elif confidence >= 0.50:
        band = "medium"
    else:
        band = "low"

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
    Deterministic sorting key:
    1. Qualifiers first
    2. No-text strictly last
    3. Scored non-qualifiers before guard/abstain
    4. Calibrated confidence descending
    5. qualify_score / ranking_score descending
    6. pub_year descending
    7. paper_id ascending
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
        0 if is_qual else 1,
        1 if is_no_text else 0,
        1 if (is_abstain or guard_rejected) else 0,
        -conf_num,
        -score_num,
        -year_val,
        paper_id,
    )


calculate_qualify_score = margin_to_qualify_score
get_confidence_band = assign_confidence_band


def rank_papers(results: list) -> list:
    return sorted(results, key=ranking_sort_key)
