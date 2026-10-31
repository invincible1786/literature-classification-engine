"""LOEO (Leave-One-Entity-Out) Cross-Validation, Threshold Tuning, and Ablation Harness."""
import os
import sys
import csv
import json
import time
import argparse
from typing import List, Dict, Any, Tuple

# Ensure base directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.classifier import classify_publications
from app.nli import DebertaNLIScorer, StubScorer
from app.text_utils import init_punkt
from eval.metrics import compute_classification_metrics, mcnemar_exact


def load_dataset() -> List[Dict[str, Any]]:
    csv_file = os.path.join(BASE_DIR, "eval", "labels.csv")
    papers = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            papers.append(r)
    return papers


def evaluate_fold(
    papers: List[Dict[str, Any]],
    tau: float,
    delta: float,
    scorer,
    two_hit_rule: bool = False,
    disable_guard: bool = False
) -> Dict[str, Any]:
    """Evaluate a set of papers under specific tau and delta parameters."""
    y_true: List[int] = []
    y_pred: List[int] = []
    fp_by_k: Dict[str, int] = {"1": 0, "2-3": 0, "4-6": 0, "7+": 0}
    total_by_k: Dict[str, int] = {"1": 0, "2-3": 0, "4-6": 0, "7+": 0}

    # Group papers by entity for classification
    by_entity: Dict[str, List[Dict[str, Any]]] = {}
    for p in papers:
        ent = p["entity"]
        by_entity.setdefault(ent, []).append(p)

    for ent, ent_papers in by_entity.items():
        # Form Europe-PMC-like dicts
        pubs = [
            {
                "title": p["title"],
                "abstractText": p["abstract"],
                "pmid": p["pmid_or_pmcid"],
                "pubYear": "2021",
                "fulltext_sections": None
            }
            for p in ent_papers
        ]
        criterion = ent_papers[0]["criterion"]

        res = classify_publications(
            publications=pubs,
            entity=ent,
            criterion=criterion,
            threshold=tau,
            contrast_delta=delta if not disable_guard else 999.0,
            scorer=scorer,
            two_hit_rule=two_hit_rule
        )

        for true_p, pred_p in zip(ent_papers, res.results):
            gold = 1 if true_p["label"] == "qualifies" else 0
            pred = 1 if pred_p["classification"] == "qualifies" else 0
            y_true.append(gold)
            y_pred.append(pred)

            k_elig = pred_p.get("k_eligible", 1)
            k_bin = "1" if k_elig <= 1 else ("2-3" if k_elig <= 3 else ("4-6" if k_elig <= 6 else "7+"))
            total_by_k[k_bin] += 1
            if gold == 0 and pred == 1:
                fp_by_k[k_bin] += 1

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["fp_by_k"] = fp_by_k
    metrics["total_by_k"] = total_by_k
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def run_loeo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true", help="Use StubScorer for fast verification")
    args = parser.parse_args()

    init_punkt()
    print("Initializing scorer...")
    scorer = StubScorer() if args.stub else DebertaNLIScorer()

    dataset = load_dataset()
    entities = sorted(list(set(p["entity"] for p in dataset)))
    print(f"Loaded {len(dataset)} papers across {len(entities)} entities: {entities}")

    # Sealed entities evaluation protocol
    # entities[:3] are training folds (vanillin, curcumin, chitin)
    # entities[3:] are held-out / sealed entities (lipase, insect_protein)
    dev_entities = entities[:3]
    sealed_entities = entities[3:]
    print(f"Dev/Tuning entities: {dev_entities}")
    print(f"Sealed evaluation entities: {sealed_entities}")

    dev_papers = [p for p in dataset if p["entity"] in dev_entities]
    sealed_papers = [p for p in dataset if p["entity"] in sealed_entities]

    # Grid search for tau and delta on Dev folds to maximize F0.5
    best_tau = 0.65
    best_delta = 0.05
    best_f05 = -1.0

    tau_candidates = [0.45, 0.55, 0.65, 0.70]
    delta_candidates = [0.00, 0.05, 0.10]

    print("\n--- Tuning Hyperparameters (tau, delta) on Dev Folds (maximizing F0.5) ---")
    for t in tau_candidates:
        for d in delta_candidates:
            res = evaluate_fold(dev_papers, tau=t, delta=d, scorer=scorer)
            f05 = res["f05"]
            p = res["precision"]
            r = res["recall"]
            print(f"  tau={t:.2f}, delta={d:.2f} -> Prec: {p:.3f}, Rec: {r:.3f}, F0.5: {f05:.3f}")
            if f05 > best_f05:
                best_f05 = f05
                best_tau = t
                best_delta = d

    print(f"\n[Optimal Parameters Selected] tau* = {best_tau:.2f}, delta* = {best_delta:.2f} (F0.5 = {best_f05:.3f})")

    # Evaluate on Sealed Entities with frozen optimal parameters
    print(f"\n--- Evaluating Frozen Model on Sealed Entities ({sealed_entities}) ---")
    final_res = evaluate_fold(sealed_papers, tau=best_tau, delta=best_delta, scorer=scorer)

    print("\n=======================================================")
    print("SEALED EVALUATION RESULTS (Frozen Parameters)")
    print("=======================================================")
    print(f"  Total Papers (N):   {final_res['n']}")
    print(f"  Accuracy:           {final_res['accuracy']:.3f}  [95% Wilson CI: {final_res['accuracy_ci'][0]:.3f} - {final_res['accuracy_ci'][1]:.3f}]")
    print(f"  Precision:          {final_res['precision']:.3f}  [95% Wilson CI: {final_res['precision_ci'][0]:.3f} - {final_res['precision_ci'][1]:.3f}]")
    print(f"  Recall:             {final_res['recall']:.3f}  [95% Wilson CI: {final_res['recall_ci'][0]:.3f} - {final_res['recall_ci'][1]:.3f}]")
    print(f"  Specificity:        {final_res['specificity']:.3f}  [95% Wilson CI: {final_res['specificity_ci'][0]:.3f} - {final_res['specificity_ci'][1]:.3f}]")
    print(f"  F1 Score:           {final_res['f1']:.3f}")
    print(f"  F0.5 Score:         {final_res['f05']:.3f}")
    print(f"  Confusion Matrix:   TP={final_res['tp']}, FP={final_res['fp']}, TN={final_res['tn']}, FN={final_res['fn']}")
    print("\nFalse Positives Stratified by Candidate Count (k_eligible):")
    for k_bin, fp_count in final_res["fp_by_k"].items():
        tot = final_res["total_by_k"][k_bin]
        pct = (fp_count / tot * 100) if tot > 0 else 0.0
        print(f"  k in {k_bin:5s}:  {fp_count}/{tot} ({pct:.1f}%)")

    # Run Ablation: Baseline V0 (No Contrast Guard) vs Final (V2: Affirmative Contrast Guard)
    v0_res = evaluate_fold(sealed_papers, tau=best_tau, delta=best_delta, scorer=scorer, disable_guard=True)
    # Discordant pairs (b, c) for McNemar
    y_true = final_res["y_true"]
    y_v2 = final_res["y_pred"]
    y_v0 = v0_res["y_pred"]

    b = sum(1 for yt, p2, p0 in zip(y_true, y_v2, y_v0) if (p2 == yt and p0 != yt))
    c = sum(1 for yt, p2, p0 in zip(y_true, y_v2, y_v0) if (p2 != yt and p0 == yt))
    mcnemar_p = mcnemar_exact(b, c)

    print(f"\nAblation Comparison (V0 Baseline without Guard vs V2 with Contrast Guard):")
    print(f"  V0 Baseline Precision: {v0_res['precision']:.3f} -> V2 Guard Precision: {final_res['precision']:.3f}")
    print(f"  Discordant Pairs: b (V2 correct, V0 wrong) = {b}, c (V2 wrong, V0 correct) = {c}")
    print(f"  Exact McNemar p-value: {mcnemar_p:.4f} (descriptive at N={len(sealed_papers)})")

    # Save evaluation summary
    out_file = os.path.join(BASE_DIR, "eval", "results_loeo.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_revision": getattr(scorer, "revision", "unknown"),
            "optimal_tau": best_tau,
            "optimal_delta": best_delta,
            "dev_entities": dev_entities,
            "sealed_entities": sealed_entities,
            "sealed_metrics": {
                "n": final_res["n"],
                "precision": round(final_res["precision"], 4),
                "recall": round(final_res["recall"], 4),
                "accuracy": round(final_res["accuracy"], 4),
                "specificity": round(final_res["specificity"], 4),
                "f1": round(final_res["f1"], 4),
                "f05": round(final_res["f05"], 4),
                "tp": final_res["tp"],
                "fp": final_res["fp"],
                "tn": final_res["tn"],
                "fn": final_res["fn"],
                "precision_ci_95": [round(x, 4) for x in final_res["precision_ci"]],
                "recall_ci_95": [round(x, 4) for x in final_res["recall_ci"]],
                "accuracy_ci_95": [round(x, 4) for x in final_res["accuracy_ci"]],
            },
            "ablation_mcnemar": {
                "b": b,
                "c": c,
                "p_value": round(mcnemar_p, 4)
            },
            "fp_by_k": final_res["fp_by_k"],
            "total_by_k": final_res["total_by_k"]
        }, f, indent=2)
    print(f"\nSaved LOEO evaluation results to {out_file}")


if __name__ == "__main__":
    run_loeo()

