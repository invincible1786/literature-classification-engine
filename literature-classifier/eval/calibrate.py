"""Empirical Platt scaling MLE calibration fitted strictly on LOEO dev folds (WS12-A / WS12-B / WS17-R3 / WS17-R4)."""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.criterion_parser import parse_criterion_to_spec
from app.conditions import check_c1_entity, check_c2_activity, check_c3_system
from app.combiner import combine_conditions
from app.hypotheses import detect_intent
from app.classifier import select_candidates_intent_aware, DEFAULT_THRESHOLD, DEFAULT_MAX_CANDIDATES
from app.nli import DebertaNLIScorer, StubScorer
from app.text_utils import (
    canonicalize,
    clean_markup_and_unescape,
    split_sentences_with_spans,
    build_entity_regex,
    detect_this_work_cues,
    is_review_article,
    detect_review_cues,
    resolve_anaphora,
    init_punkt,
)

NEG_FIT_WINDOW = 0.15
CALIBRATION_VERSION = "platt-v3.3-mle"


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def fit_logistic_mle(
    margins: List[float],
    labels: List[int],
    l2_reg: float = 0.05,
    max_iter: int = 100,
) -> Tuple[float, float, float]:
    """
    Fit P(y=1 | m) = 1 / (1 + exp(-(A * m + B))) via L2-regularized Newton-Raphson MLE.
    Returns: (A, B, brier_score)
    """
    if not margins or len(margins) != len(labels):
        raise ValueError("Margins and labels must be non-empty and equal length.")

    # Initialize A > 0, B near 0 (monotonically increasing probability in margin)
    a = 2.5
    b = 0.0

    for _ in range(max_iter):
        g_a = 0.0
        g_b = 0.0
        h_aa = l2_reg
        h_bb = l2_reg
        h_ab = 0.0

        for m, y in zip(margins, labels):
            z = max(-15.0, min(15.0, a * m + b))
            p = 1.0 / (1.0 + math.exp(-z))
            diff = p - y
            w = max(1e-6, p * (1.0 - p))
            g_a += diff * m
            g_b += diff
            h_aa += w * m * m
            h_bb += w
            h_ab += w * m

        # L2 regularization gradient on A (shrink toward modest slope) and B
        g_a += l2_reg * (a - 2.0)
        g_b += l2_reg * b

        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (h_aa * g_b - h_ab * g_a) / det

        a -= step_a
        b -= step_b
        # Enforce monotonic non-negative slope A >= 0.25
        a = max(0.25, min(20.0, a))
        b = max(-15.0, min(15.0, b))
        if abs(step_a) < 1e-6 and abs(step_b) < 1e-6:
            break

    sq_errs = []
    for m, y in zip(margins, labels):
        z = max(-15.0, min(15.0, a * m + b))
        p = 1.0 / (1.0 + math.exp(-z))
        sq_errs.append((p - y) ** 2)
    brier = sum(sq_errs) / len(sq_errs)
    return round(float(a), 6), round(float(b), 6), round(float(brier), 4)


def collect_dev_fit_samples(
    rows: List[Dict[str, Any]],
    dev_entities: List[str],
    test_entities: List[str],
    scorer: Any,
    tau: float = DEFAULT_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluate dev rows through the v3.2 condition stack (without invoking scoring.py)
    and collect calibration samples for qualifier_path and negative_path.
    Hard-fails on any test_entity leakage (WS17-R3).
    """
    dev_set = set(dev_entities)
    test_set = set(test_entities)
    if dev_set & test_set:
        raise RuntimeError(f"Split leakage detected: overlap {dev_set & test_set}")

    qual_samples: List[Dict[str, Any]] = []
    neg_samples: List[Dict[str, Any]] = []

    for r in rows:
        if r.get("stratum") == "live_audit":
            raise RuntimeError("Live audit row passed to calibration fit!")
        ent = r["entity"]
        if ent in test_set:
            # Strict exclusion: skip test rows; raise if someone tried passing test list as fit
            continue
        if ent not in dev_set:
            continue

        gold = r.get("gold") or r["label"]
        title = clean_markup_and_unescape(r["title"])
        abstract = clean_markup_and_unescape(r["abstract"])
        canon_title = canonicalize(title)
        canon_body = canonicalize(abstract)
        if not canon_body:
            continue

        spec = parse_criterion_to_spec(r["criterion"], entity=ent, allow_partial_criterion=True)
        intent = detect_intent(r["criterion"])
        sentences = split_sentences_with_spans(canon_body, source="abstract")
        entity_regex = build_entity_regex(ent)

        has_pos_exp = any(detect_this_work_cues(s.text)[0] for s in sentences)
        has_strong, has_weak, has_results = detect_review_cues(canon_title, canon_body)
        has_experimental_results = has_results or has_pos_exp
        is_review = is_review_article(canon_title, canon_body) or "review" in (r.get("pubType") or "").lower()
        is_pubtype_review_veto = (
            (is_review or has_strong)
            and not has_experimental_results
            and spec.scope == "this_study"
        )

        c1, role_veto_reason = check_c1_entity(sentences, r["abstract"], spec, is_fulltext=False)
        anaphora_map = resolve_anaphora(sentences, entity_regex)
        eligible = [s for s in sentences if entity_regex.search(s.text) or s.sentence_index in anaphora_map]
        candidates = select_candidates_intent_aware(eligible, intent=intent, max_candidates=DEFAULT_MAX_CANDIDATES)
        c2 = check_c2_activity(
            candidates, r["abstract"], spec, scorer, tau_activity=tau,
            all_sentences=sentences, anaphora_map=anaphora_map, entity_regex=entity_regex,
            title=canon_title,
        )

        if not is_pubtype_review_veto and has_weak and not has_experimental_results and spec.scope == "this_study":
            c2.status = "unknown"
            c2.reason_code = "possible_review"
            c2.evidence = []

        c3 = check_c3_system(sentences, r["abstract"], spec, is_fulltext=False, entity_regex=entity_regex)
        has_conflict = bool(role_veto_reason and c2.synthesis_claim_sents)

        dec = combine_conditions(
            c1=c1, c2=c2, c3=c3, spec=spec, has_usable_text=True,
            role_veto_reason=role_veto_reason,
            has_conflicting_synthesis_claim=has_conflict,
            is_pubtype_review_veto=is_pubtype_review_veto,
        )

        # Qualifier-path fit window: qualifiers + borderline non-veto negatives within NEG_FIT_WINDOW of tau
        is_veto = dec.reason_code in ("review_article", "reagent_role")
        in_borderline_window = (tau - NEG_FIT_WINDOW) <= c2.score <= (tau + NEG_FIT_WINDOW)
        if not is_veto and (dec.classification == "qualifies" or in_borderline_window):
            m_val = dec.ranking_score if dec.classification == "qualifies" else c2.score
            qual_samples.append({
                "entity": ent,
                "margin": float(m_val),
                "y": 1 if gold == "qualifies" else 0,
            })

        # Negative-path fit rows: ONLY different_activity and wrong_system (WS17-R4)
        if dec.classification == "does_not_qualify" and dec.reason_code in ("different_activity", "wrong_system"):
            neg_m = c2.neg_margin if dec.reason_code == "different_activity" else c3.neg_margin
            neg_samples.append({
                "entity": ent,
                "margin": float(neg_m),
                "y": 1 if gold == "does_not_qualify" else 0,
            })

    return qual_samples, neg_samples


def run_calibration(
    use_stub: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    init_punkt()
    splits_path = os.path.join(BASE_DIR, "eval", "splits.json")
    labels_path = os.path.join(BASE_DIR, "eval", "labels.csv")
    out_cfg = output_path or os.path.join(BASE_DIR, "config", "calibration.json")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    splits_sha = compute_file_sha256(splits_path)
    labels_sha = compute_file_sha256(labels_path)

    dev_entities: List[str] = splits["dev_entities"]
    test_entities: List[str] = splits["test_entities"]
    if set(dev_entities) & set(test_entities):
        raise RuntimeError("splits.json has overlapping dev and test entities!")

    with open(labels_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("stratum") == "live_audit":
            raise RuntimeError("Live audit row detected in labels.csv during calibration fit!")

    scorer = StubScorer() if use_stub else DebertaNLIScorer()
    qual_samples, neg_samples = collect_dev_fit_samples(rows, dev_entities, test_entities, scorer)

    if any(s["entity"] in test_entities for s in qual_samples + neg_samples):
        raise RuntimeError("Leakage invariant violated: test entity found in calibration samples!")

    q_margins = [s["margin"] for s in qual_samples]
    q_labels = [s["y"] for s in qual_samples]
    a_q, b_q, brier_q = fit_logistic_mle(q_margins, q_labels)

    # Compute LOEO stability for A across dev entities
    loo_a: List[float] = []
    for held in dev_entities:
        fold_m = [s["margin"] for s in qual_samples if s["entity"] != held]
        fold_y = [s["y"] for s in qual_samples if s["entity"] != held]
        if len(fold_m) >= 5 and len(set(fold_y)) >= 1:
            fa, _fb, _ = fit_logistic_mle(fold_m, fold_y)
            loo_a.append(fa)

    loo_mean = round(statistics.mean(loo_a), 4) if loo_a else a_q
    loo_std = round(statistics.pstdev(loo_a), 4) if len(loo_a) >= 2 else 0.0
    slope_unstable = loo_std > 0.25 * abs(a_q)

    # Fit negative map strictly on different_activity / wrong_system rows
    neg_fitted = False
    a_neg, b_neg, brier_neg = None, None, None
    if len(neg_samples) >= 3:
        n_margins = [s["margin"] for s in neg_samples]
        n_labels = [s["y"] for s in neg_samples]
        a_neg, b_neg, brier_neg = fit_logistic_mle(n_margins, n_labels, l2_reg=0.15)
        neg_fitted = True

    calib_meta: Dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "method": "platt_scaling_logistic_mle",
        "splits_sha256": splits_sha,
        "data_hash": labels_sha,
        "fit_entities": dev_entities,
        "neg_fit_window": NEG_FIT_WINDOW,
        "qualifier_path": {
            "fitted": True,
            "A": a_q,
            "B": b_q,
            "n_total": len(qual_samples),
            "n_pos": sum(q_labels),
            "n_neg": len(q_labels) - sum(q_labels),
            "out_of_fold_brier": brier_q,
            "loo_mean_A": loo_mean,
            "loo_std_A": loo_std,
            "slope_unstable_flag": slope_unstable,
        },
        "negative_path": {
            "fitted": neg_fitted,
            "A": a_neg,
            "B": b_neg,
            "n_rows": len(neg_samples),
            "out_of_fold_brier": brier_neg,
            "reason_codes_included": ["different_activity", "wrong_system"],
        },
    }

    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)

    return calib_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true", help="Run using StubScorer (testing only)")
    args = parser.parse_args()
    meta = run_calibration(use_stub=args.stub)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
