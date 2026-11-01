from typing import Optional, List, Dict, Any

# Heuristic cut-offs for confidence bands
CONF_HIGH_THRESHOLD = 0.8
CONF_MED_THRESHOLD = 0.6


def calculate_qualify_score(m: float) -> float:
    """
    Computes ranking score qualify_score = (m + 1) / 2 in [0, 1].
    Derived strictly from the best sentence margin m in [-1, 1].
    """
    score = (m + 1.0) / 2.0
    return round(max(0.0, min(1.0, score)), 4)


def calculate_confidence(
    m: float,
    tau: float,
    is_qualifies: bool,
    contrast_flipped: bool = False,
    c_best: float = 0.0
) -> float:
    """
    Computes symmetric confidence as distance from decision boundary tau:
    - If qualifies: conf = 0.5 + 0.5 * (m - tau) / (1 - tau)
    - If does_not_qualify: conf = 0.5 + 0.5 * (tau - m) / (tau + 1)
    - If contrast guard flipped decision: The paper had m >= tau but was flipped by contrast guard.
      Rather than mixing raw softmax probability with distance-from-boundary, we assign a bounded
      moderate confidence (capped at 0.65) to faithfully reflect a guarded/conflicting decision.
    Clamped to [0.5, 1.0].
    """
    if is_qualifies:
        denom = 1.0 - tau
        if denom <= 1e-6:
            conf = 1.0 if m >= tau else 0.5
        else:
            conf = 0.5 + 0.5 * ((m - tau) / denom)
    else:
        if contrast_flipped:
            conf = 0.65
        else:
            denom = tau + 1.0
            if denom <= 1e-6:
                conf = 0.5
            else:
                conf = 0.5 + 0.5 * ((tau - m) / denom)

    conf = max(0.5, min(1.0, conf))
    return round(conf, 4)


def get_confidence_band(
    conf: Optional[float],
    is_insufficient: bool = False,
    margin_gap: Optional[float] = None
) -> str:
    """
    Assigns confidence band:
    - 'insufficient' if insufficient_text or conf is None
    - 'high' if conf >= 0.8 (and runner-up margin gap >= 0.10 if competing candidates exist)
    - 'medium' if 0.6 <= conf < 0.8 (or high conf capped due to close runner-up sentence ambiguity)
    - 'low' if conf < 0.6
    """
    if is_insufficient or conf is None:
        return "insufficient"
    if conf >= CONF_HIGH_THRESHOLD:
        if margin_gap is not None and margin_gap < 0.10:
            return "medium"
        return "high"
    if conf >= CONF_MED_THRESHOLD:
        return "medium"
    return "low"


def rank_papers(
    paper_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Sort papers:
    - Usable text papers first, insufficient_text papers strictly last.
    - Among usable text: qualify_score descending.
    - Tiebreak 1: entity present in title (True before False).
    - Tiebreak 2: number of entity sentences (descending).
    Note: Title and multi-mention features are deliberately excluded from the decision signal m
    to ensure ranking reflects pure semantic entailment, and are used solely as deterministic tiebreakers.
    """
    def sort_key(item: Dict[str, Any]):
        is_insufficient = item.get("evidence_status") == "insufficient_text"
        qs = item.get("qualify_score")
        qualify_score = qs if qs is not None else -1.0
        title_has_entity = 1 if item.get("title_has_entity", False) else 0
        n_entity_sentences = item.get("n_entity_sentences", 0)

        # In Python sort: False < True, so not is_insufficient puts usable papers first
        return (
            not is_insufficient,
            qualify_score,
            title_has_entity,
            n_entity_sentences
        )

    return sorted(paper_results, key=sort_key, reverse=True)

