"""Reproducible 3-class evaluation harness for v2 baseline vs v3.2 upgrade (WS13-B/C/D / WS17-R3/R6)."""
import argparse
import csv
import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.classifier import classify_publications, DEFAULT_THRESHOLD, DEFAULT_CONTRAST_DELTA
from app.nli import DebertaNLIScorer, StubScorer
from app.scoring import load_calibration_config, reload_calibration_config
from app.text_utils import init_punkt, compute_sha256
from eval.metrics import (
    compute_classification_metrics,
    compute_multiclass_metrics,
    compute_ece,
    cohen_kappa,
    mcnemar_exact,
)


def load_splits() -> Dict[str, Any]:
    path = os.path.join(BASE_DIR, "eval", "splits.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(stratum_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load benchmark dataset with stratum isolation (§0.5 contract)."""
    if stratum_filter == "live_audit":
        csv_file = os.path.join(BASE_DIR, "eval", "live_audit_labels.csv")
        papers = []
        with open(csv_file, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                papers.append(r)
        return papers

    csv_file = os.path.join(BASE_DIR, "eval", "labels.csv")
    papers = []
    with open(csv_file, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            strat = r.get("stratum", "clean")
            if strat not in ("clean", "noisy"):
                continue
            if stratum_filter and stratum_filter != "all" and strat != stratum_filter:
                continue
            papers.append(r)
    return papers


def audit_snapshot_manifest(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify every dataset row against data/snapshots/manifest.json SHA-256 digests (WS13-B)."""
    manifest_path = os.path.join(BASE_DIR, "data", "snapshots", "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"Missing snapshot manifest at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest.get("records", {})
    live_records = manifest.get("live_audit_records", {})
    mismatches: List[str] = []
    audited = 0
    for r in rows:
        key = f"{r['entity']}::{r['pmid_or_pmcid']}"
        rec = live_records.get(key) if r.get("stratum") == "live_audit" else records.get(key)
        if not rec:
            rec = records.get(key) or live_records.get(key)
        if not rec:
            mismatches.append(f"Missing manifest entry for {key}")
            continue
        raw_rec = f"{r.get('title') or ''} {r.get('abstract') or ''}".strip()
        actual_sha = compute_sha256(raw_rec)
        if actual_sha != rec["sha256"]:
            mismatches.append(f"SHA256 drift for {key}: expected {rec['sha256']}, got {actual_sha}")
        audited += 1
    if mismatches:
        raise RuntimeError("Snapshot SHA-256 drift detected:\n" + "\n".join(mismatches[:10]))
    return {"audited_records": audited, "manifest_version": manifest.get("manifest_version")}


def compute_double_label_kappa(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Cohen's kappa on double-labeled noisy/clean overlap."""
    r1_bin, r2_bin = [], []
    exact_matches = 0
    for r in rows:
        sec = (r.get("second_label") or "").strip()
        if not sec:
            continue
        gold = r.get("gold") or r["label"]
        if gold == sec:
            exact_matches += 1
        r1_bin.append(1 if gold == "qualifies" else 0)
        r2_bin.append(1 if sec == "qualifies" else 0)
    kappa = cohen_kappa(r1_bin, r2_bin) if r1_bin else 0.0
    return {
        "n_double_labeled": len(r1_bin),
        "exact_agreement": round(exact_matches / len(r1_bin), 4) if r1_bin else 0.0,
        "cohens_kappa": round(kappa, 4),
    }


def load_pubchem_cache() -> Dict[str, List[str]]:
    path = os.path.join(BASE_DIR, "data", "snapshots", "pubchem_cache.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {k.lower(): v for k, v in json.load(f).items()}
    return {}


def evaluate_rows(
    rows: List[Dict[str, Any]],
    scorer: Any,
    pipeline: str = "v3",
    tau: float = DEFAULT_THRESHOLD,
    delta: float = DEFAULT_CONTRAST_DELTA,
    strict_verification: bool = True,
) -> Dict[str, Any]:
    """Evaluate a set of benchmark rows and return 3-class metrics, ECE, and per-paper traces."""
    alias_cache = load_pubchem_cache()
    by_entity: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_entity.setdefault(r["entity"], []).append(r)

    traces: List[Dict[str, Any]] = []
    y_true_3: List[str] = []
    y_pred_3: List[str] = []
    cal_probs: List[float] = []
    cal_corrects: List[int] = []
    raw_probs: List[float] = []
    raw_corrects: List[int] = []

    for ent, ent_rows in by_entity.items():
        ent_aliases = alias_cache.get(ent.lower(), [])
        pubs = [
            {
                "title": r["title"],
                "abstractText": r["abstract"],
                "pmid": r["pmid_or_pmcid"],
                "pubYear": "2025",
                "pubType": r.get("pubType", ""),
                "fulltext_sections": None,
            }
            for r in ent_rows
        ]
        criterion = ent_rows[0]["criterion"]

        # Map individual criteria rows cleanly if criterion differs within same entity
        by_crit: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        for i_row, r in enumerate(ent_rows):
            by_crit.setdefault(r["criterion"], []).append((i_row, r))

        preds_by_idx: Dict[int, Dict[str, Any]] = {}
        for crit_str, items in by_crit.items():
            sub_pubs = [pubs[i_row] for i_row, _ in items]
            run_res = classify_publications(
                publications=sub_pubs,
                entity=ent,
                criterion=crit_str,
                aliases=ent_aliases,
                expand_aliases=False,
                tau=tau,
                delta=delta,
                scorer=scorer,
                pipeline=pipeline,
                strict_verification=strict_verification,
            )
            # Re-align ranked results back by paper_id
            by_pid = {str(p["paper_id"]): p for p in run_res.results}
            for i_row, r in items:
                preds_by_idx[i_row] = by_pid[str(r["pmid_or_pmcid"])]

        for i_row, r in enumerate(ent_rows):
            pred_obj = preds_by_idx[i_row]
            gold = r.get("gold") or r["label"]
            pred_cls = pred_obj["classification"]
            if pipeline == "v2" and pred_cls not in ("qualifies", "does_not_qualify"):
                pred_cls = "does_not_qualify"

            y_true_3.append(gold)
            y_pred_3.append(pred_cls)

            conf_obj = pred_obj.get("confidence")
            conf_score = None
            if isinstance(conf_obj, dict):
                conf_score = conf_obj.get("score")
            elif hasattr(conf_obj, "score"):
                conf_score = conf_obj.score
            elif isinstance(conf_obj, (int, float)):
                conf_score = float(conf_obj)

            # Calibration ECE tracking on confident outputs (qualifies + calibrated negatives)
            if conf_score is not None:
                is_corr = 1 if pred_cls == gold else 0
                cal_probs.append(float(conf_score))
                cal_corrects.append(is_corr)

            # Baseline pre-calibration raw margin probability for before/after ECE comparison
            q_raw = pred_obj.get("qualify_score")
            if pred_cls == "qualifies":
                raw_score = (pred_obj.get("ranking_score", 0.5) + 1.0) / 2.0
                raw_probs.append(max(0.0, min(1.0, float(raw_score))))
                raw_corrects.append(1 if gold == "qualifies" else 0)
            elif q_raw is not None:
                raw_probs.append(max(0.0, min(1.0, float(q_raw))))
                raw_corrects.append(1 if pred_cls == gold else 0)

            traces.append({
                "entity": ent,
                "paper_id": r["pmid_or_pmcid"],
                "stratum": r.get("stratum", "clean"),
                "category": r.get("category", ""),
                "gold": gold,
                "pred": pred_cls,
                "reason_code": pred_obj.get("reason_code"),
                "confidence": conf_score,
                "evidence": pred_obj.get("evidence_sentence"),
                "conditions": pred_obj.get("conditions"),
            })

    mc = compute_multiclass_metrics(y_true_3, y_pred_3)
    ece_cal, ece_bins = compute_ece(cal_probs, cal_corrects)
    ece_raw, _ = compute_ece(raw_probs, raw_corrects)

    # Stratified clean vs noisy
    strata_metrics: Dict[str, Any] = {}
    for st in ("clean", "noisy"):
        sub_t = [t["gold"] for t in traces if t["stratum"] == st]
        sub_p = [t["pred"] for t in traces if t["stratum"] == st]
        if sub_t:
            strata_metrics[st] = compute_multiclass_metrics(sub_t, sub_p)

    # Binary qualifier slice metrics
    y_true_bin = [1 if t == "qualifies" else 0 for t in y_true_3]
    y_pred_bin = [1 if p == "qualifies" else 0 for p in y_pred_3]
    bin_metrics = compute_classification_metrics(y_true_bin, y_pred_bin)

    return {
        "pipeline": pipeline,
        "multiclass": mc,
        "binary_qualifies_slice": bin_metrics,
        "by_stratum": strata_metrics,
        "ece_calibrated": ece_cal,
        "ece_raw_baseline": ece_raw,
        "ece_bins": ece_bins,
        "y_true_3": y_true_3,
        "y_pred_3": y_pred_3,
        "traces": traces,
    }


def run_eval_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["v2", "v3", "both"], default="both")
    parser.add_argument("--split", choices=["test", "dev", "all"], default="all")
    parser.add_argument("--stratum", choices=["all", "clean", "noisy", "live_audit"], default="all")
    parser.add_argument("--stub", action="store_true", help="Use StubScorer (testing only)")
    parser.add_argument("--strict-splits", action="store_true", default=True)
    args = parser.parse_args()

    init_punkt()
    splits = load_splits()
    rows = load_dataset(stratum_filter=args.stratum)
    audit_info = audit_snapshot_manifest(rows)

    if args.stratum == "live_audit":
        target_ents = {r["entity"] for r in rows}
    elif args.split == "dev":
        target_ents = set(splits["dev_entities"])
    elif args.split == "test":
        target_ents = set(splits["test_entities"])
    else:
        target_ents = set(splits["dev_entities"]) | set(splits["test_entities"])

    eval_rows = [r for r in rows if r["entity"] in target_ents]
    reload_calibration_config()
    load_calibration_config(strict_splits=args.strict_splits)

    scorer = StubScorer() if args.stub else DebertaNLIScorer()

    kappa_noisy = compute_double_label_kappa([r for r in eval_rows if r.get("stratum") == "noisy"])
    kappa_all = compute_double_label_kappa(eval_rows)

    out_dir = os.path.join(BASE_DIR, "eval")
    res_v2 = None
    res_v3 = None

    if args.pipeline in ("v2", "both"):
        res_v2 = evaluate_rows(eval_rows, scorer=scorer, pipeline="v2")
        with open(os.path.join(out_dir, "results_v2.json"), "w", encoding="utf-8") as f:
            json.dump(res_v2, f, indent=2)

    if args.pipeline in ("v3", "both"):
        res_v3 = evaluate_rows(eval_rows, scorer=scorer, pipeline="v3")
        with open(os.path.join(out_dir, "results_v3.json"), "w", encoding="utf-8") as f:
            json.dump(res_v3, f, indent=2)

    comparison: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": args.split,
        "stratum": args.stratum,
        "snapshot_audit": audit_info,
        "inter_annotator_noisy": kappa_noisy,
        "inter_annotator_all": kappa_all,
    }
    if res_v2 and res_v3:
        yt = res_v3["y_true_3"]
        p3 = res_v3["y_pred_3"]
        p2 = res_v2["y_pred_3"]
        b_3 = sum(1 for t, c3, c2 in zip(yt, p3, p2) if c3 == t and c2 != t)
        c_3 = sum(1 for t, c3, c2 in zip(yt, p3, p2) if c3 != t and c2 == t)
        p_val_3 = mcnemar_exact(b_3, c_3)

        # Binary qualifier sub-test
        yb = [1 if t == "qualifies" else 0 for t in yt]
        b3 = [1 if p == "qualifies" else 0 for p in p3]
        b2 = [1 if p == "qualifies" else 0 for p in p2]
        b_bin = sum(1 for t, c3, c2 in zip(yb, b3, b2) if c3 == t and c2 != t)
        c_bin = sum(1 for t, c3, c2 in zip(yb, b3, b2) if c3 != t and c2 == t)
        p_val_bin = mcnemar_exact(b_bin, c_bin)

        comparison["mcnemar_3class"] = {"b": b_3, "c": c_3, "p_value": round(p_val_3, 4)}
        comparison["mcnemar_binary_qualifies"] = {"b": b_bin, "c": c_bin, "p_value": round(p_val_bin, 4)}
        comparison["v2_macro_f1"] = round(res_v2["multiclass"]["macro_f1"], 4)
        comparison["v3_macro_f1"] = round(res_v3["multiclass"]["macro_f1"], 4)
        comparison["v2_qualifies_fp"] = res_v2["multiclass"]["qualifies_fp"]
        comparison["v3_qualifies_fp"] = res_v3["multiclass"]["qualifies_fp"]
        comparison["v3_abstention_rate"] = round(res_v3["multiclass"]["abstention_rate"], 4)
        comparison["v3_abstention_error_rate"] = round(res_v3["multiclass"]["abstention_error_rate"], 4)
        comparison["v3_ece_calibrated"] = res_v3["ece_calibrated"]
        comparison["v2_ece_raw"] = res_v2["ece_raw_baseline"]

    with open(os.path.join(out_dir, "results_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    run_eval_cli()
