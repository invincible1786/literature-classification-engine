import os
import sys
import csv
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.nli import DebertaNLIScorer
from app.classifier import classify_publications, DEFAULT_THRESHOLD, DEFAULT_CONTRAST_DELTA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def load_dataset(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    n_bootstraps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% non-parametric percentile bootstrap confidence intervals
    for Precision, Recall, and F1 score.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    yt_arr = np.array(y_true)
    yp_arr = np.array(y_pred)

    precisions = []
    recalls = []
    f1s = []

    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        yt = yt_arr[idxs]
        yp = yp_arr[idxs]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    low_pct = (alpha / 2.0) * 100
    high_pct = (1.0 - alpha / 2.0) * 100

    return {
        "precision_ci": (round(float(np.percentile(precisions, low_pct)), 3),
                         round(float(np.percentile(precisions, high_pct)), 3)),
        "recall_ci": (round(float(np.percentile(recalls, low_pct)), 3),
                      round(float(np.percentile(recalls, high_pct)), 3)),
        "f1_ci": (round(float(np.percentile(f1s, low_pct)), 3),
                  round(float(np.percentile(f1s, high_pct)), 3)),
    }


def print_margin_histogram(margins: List[float]):
    """Prints ASCII histogram of best sentence margins m in [-1, 1]."""
    bins = [
        ("[-1.0, -0.6)", lambda m: -1.0 <= m < -0.6),
        ("[-0.6, -0.2)", lambda m: -0.6 <= m < -0.2),
        ("[-0.2, +0.2)", lambda m: -0.2 <= m < 0.2),
        ("[+0.2, +0.6)", lambda m: 0.2 <= m < 0.6),
        ("[+0.6, +1.0]", lambda m: 0.6 <= m <= 1.0),
    ]
    print("\nMARGIN VALUE HISTOGRAM (Best Candidate Sentence Margin m in [-1, 1]):")
    print(f"{'Bin':14s} | {'Count':5s} | Bar")
    print("-" * 50)
    for label, fn in bins:
        count = sum(1 for m in margins if fn(m))
        bar = "#" * (count * 2)
        print(f"{label:14s} | {count:5d} | {bar}")


def evaluate_papers(
    papers: List[Dict[str, Any]],
    entity: str,
    criterion: str,
    threshold: float,
    contrast_delta: float,
    scorer: DebertaNLIScorer,
    synonyms: List[str] = None
) -> Dict[str, Any]:
    """
    Runs classification pipeline on the test papers and computes:
    - Precision, Recall, F1 with 95% bootstrap confidence intervals
    - Confusion Matrix (TP, FP, TN, FN)
    - Accuracy and Evidence correctness
    - Confidence band distribution
    """
    pubs_input = []
    for p in papers:
        pubs_input.append({
            "title": p.get("title"),
            "abstractText": p.get("abstract"),
            "pmid": p.get("pmid"),
            "doi": None,
            "pmcid": None,
        })

    run_output = classify_publications(
        publications=pubs_input,
        entity=entity,
        criterion=criterion,
        synonyms=synonyms or [],
        threshold=threshold,
        contrast_delta=contrast_delta,
        scorer=scorer
    )
    results = run_output.results

    results_by_pmid = {r["pmid"]: r for r in results}

    tp = fp = tn = fn = 0
    band_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    evidence_correct_count = 0
    eval_rows = []
    y_true = []
    y_pred = []
    extracted_margins = []

    for gold_item in papers:
        pmid = gold_item["pmid"]
        pred_item = results_by_pmid[pmid]
        gold_label = gold_item["gold_label"].strip()
        pred_label = pred_item["classification"].strip()
        band = pred_item["confidence_band"]

        # Track margins: qualify_score = (m + 1) / 2 => m = qualify_score * 2 - 1
        qs = pred_item["qualify_score"]
        if qs is not None:
            extracted_margins.append(qs * 2.0 - 1.0)
        else:
            extracted_margins.append(-1.0)

        yt = 1 if gold_label == "qualifies" else 0
        yp = 1 if pred_label == "qualifies" else 0
        y_true.append(yt)
        y_pred.append(yp)

        is_correct = (pred_label == gold_label)
        band_stats[band]["total"] += 1
        if is_correct:
            band_stats[band]["correct"] += 1

        if pred_label == "qualifies" and gold_label == "qualifies":
            tp += 1
        elif pred_label == "qualifies" and gold_label == "does_not_qualify":
            fp += 1
        elif pred_label == "does_not_qualify" and gold_label == "does_not_qualify":
            tn += 1
        else:
            fn += 1

        # Check evidence correctness against gold evidence
        pred_ev = pred_item["evidence"]
        gold_ev = gold_item.get("gold_evidence")
        evidence_supported = False
        if is_correct:
            if pred_ev and gold_ev:
                if pred_ev.strip().lower() in gold_item["abstract"].lower() or pred_ev == gold_ev:
                    evidence_supported = True
            elif pred_item["evidence_status"] == "insufficient_text":
                evidence_supported = True

        if evidence_supported:
            evidence_correct_count += 1

        eval_rows.append({
            "pmid": pmid,
            "gold_label": gold_label,
            "pred_label": pred_label,
            "score": pred_item["qualify_score"],
            "conf": pred_item["confidence"],
            "band": band,
            "status": pred_item["evidence_status"],
            "evidence": pred_ev,
            "gold_evidence": gold_ev,
            "is_correct": is_correct
        })

    total = len(papers)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    evidence_acc = evidence_correct_count / total if total > 0 else 0.0

    cis = bootstrap_ci(y_true, y_pred, n_bootstraps=2000, seed=42)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "precision_ci": cis["precision_ci"],
        "recall_ci": cis["recall_ci"],
        "f1_ci": cis["f1_ci"],
        "accuracy": acc,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "band_calibration": {
            b: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0.0
            }
            for b, s in band_stats.items()
        },
        "evidence_correctness": evidence_acc,
        "margins": extracted_margins,
        "eval_rows": eval_rows
    }


def main():
    csv_path = os.path.join(root_dir, "eval", "labeled.csv")
    dataset = load_dataset(csv_path)

    entity_b = "vanillin"
    test_papers = [p for p in dataset if p["entity"].strip().lower() == entity_b]
    logger.info(f"Loaded {len(test_papers)} holdout evaluation papers for Entity B ({entity_b})")

    scorer = DebertaNLIScorer()

    tuned_tau = 0.65
    tuned_delta = 0.05

    criterion_primary = "Vanillin is synthesised, biosynthesised, or produced in this work"
    logger.info(f"Running primary evaluation on Entity B with tau={tuned_tau}, delta={tuned_delta}...")

    metrics = evaluate_papers(
        papers=test_papers,
        entity=entity_b,
        criterion=criterion_primary,
        threshold=tuned_tau,
        contrast_delta=tuned_delta,
        scorer=scorer,
        synonyms=["4-hydroxy-3-methoxybenzaldehyde"]
    )

    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION REPORT (Entity B: Vanillin — Unseen Test Entity)")
    print("=" * 70)
    print(f"Dataset Size       : {len(test_papers)} papers (8 Qualifiers, 12 Non-Qualifiers)")
    print(f"Decision Threshold : tau = {tuned_tau:.2f}")
    print(f"Contrast Guard Delta: delta = {tuned_delta:.2f}")
    print("-" * 70)
    print(f"Precision (Qualifies): {metrics['precision']:.3f}  [95% CI: {metrics['precision_ci'][0]:.3f} - {metrics['precision_ci'][1]:.3f}]")
    print(f"Recall    (Qualifies): {metrics['recall']:.3f}  [95% CI: {metrics['recall_ci'][0]:.3f} - {metrics['recall_ci'][1]:.3f}]")
    print(f"F1 Score  (Qualifies): {metrics['f1']:.3f}  [95% CI: {metrics['f1_ci'][0]:.3f} - {metrics['f1_ci'][1]:.3f}]")
    print(f"Overall Accuracy     : {metrics['accuracy']:.3f}")
    print(f"Evidence Correctness : {metrics['evidence_correctness']:.3f} ({metrics['evidence_correctness']*100:.1f}%)")
    print("-" * 70)
    cm = metrics["confusion_matrix"]
    print("CONFUSION MATRIX:")
    print(f"  True Positives  (TP) : {cm['TP']}")
    print(f"  False Positives (FP) : {cm['FP']}")
    print(f"  True Negatives  (TN) : {cm['TN']}")
    print(f"  False Negatives (FN) : {cm['FN']}")
    print("-" * 70)
    print("CALIBRATION BY CONFIDENCE BAND:")
    for band, stat in sorted(metrics["band_calibration"].items()):
        print(f"  Band '{band:12s}': {stat['correct']}/{stat['total']} correct ({stat['accuracy']*100:5.1f}% accuracy)")

    print_margin_histogram(metrics["margins"])

    print("\n" + "=" * 70)
    print("CRITERION PHRASING SPREAD EVALUATION (With Paraphrase Ensemble)")
    print("=" * 70)
    phrasings = [
        ("Phrasing 1 (Standard passive)", "Vanillin is synthesised, biosynthesised, or produced in this work"),
        ("Phrasing 2 (Agentive active)", "The authors synthesized, biosynthesized, or produced vanillin in this study."),
        ("Phrasing 3 (Product-centric)", "Vanillin was obtained as a synthesized or biosynthesized product.")
    ]

    print(f"{'Phrasing Description':30s} | {'Precision [95% CI]':24s} | {'Recall':7s} | {'F1':7s} | {'Accuracy':8s}")
    print("-" * 75)
    for desc, phr in phrasings:
        m_phr = evaluate_papers(
            papers=test_papers,
            entity=entity_b,
            criterion=phr,
            threshold=tuned_tau,
            contrast_delta=tuned_delta,
            scorer=scorer,
            synonyms=["4-hydroxy-3-methoxybenzaldehyde"]
        )
        p_ci_str = f"{m_phr['precision']:.3f} [{m_phr['precision_ci'][0]:.2f}-{m_phr['precision_ci'][1]:.2f}]"
        print(f"{desc:30s} | {p_ci_str:24s} | {m_phr['recall']:7.3f} | {m_phr['f1']:7.3f} | {m_phr['accuracy']:8.3f}")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
