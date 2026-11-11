"""Statistical evaluation metrics for literature classification."""
import math
from typing import Tuple, Dict, Any, List

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a binomial proportion.
    
    Args:
        k: Number of successes / positives.
        n: Total trials.
        z: Z-score for confidence level (1.96 for 95%).
        
    Returns:
        (lower_bound, upper_bound)
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + (z * z) / n
    c = (p + (z * z) / (2.0 * n)) / d
    h = z * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def mcnemar_exact(b: int, c: int) -> float:
    """Exact binomial two-tailed McNemar test using math.comb.
    
    Args:
        b: Number of cases where Model 1 is correct and Model 2 is incorrect.
        c: Number of cases where Model 1 is incorrect and Model 2 is correct.
        
    Returns:
        Exact two-tailed p-value.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * p)

def cohen_kappa(r1: List[int], r2: List[int]) -> float:
    """Calculate Cohen's kappa for two raters with binary labels (0 and 1)."""
    if len(r1) != len(r2) or len(r1) == 0:
        return 0.0
    n = len(r1)
    # Confusion matrix:
    # a: both 1, b: r1=1, r2=0, c: r1=0, r2=1, d: both 0
    a = sum(1 for x, y in zip(r1, r2) if x == 1 and y == 1)
    b = sum(1 for x, y in zip(r1, r2) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(r1, r2) if x == 0 and y == 1)
    d = sum(1 for x, y in zip(r1, r2) if x == 0 and y == 0)
    
    po = (a + d) / n
    p1 = ((a + b) / n) * ((a + c) / n)
    p0 = ((c + d) / n) * ((b + d) / n)
    pe = p1 + p0
    
    if abs(1.0 - pe) < 1e-9:
        return 1.0
    return (po - pe) / (1.0 - pe)

def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Compute standard classification metrics with Wilson confidence intervals."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
    
    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "accuracy_ci": wilson(tp + tn, total),
        "precision": precision,
        "precision_ci": wilson(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0),
        "recall": recall,
        "recall_ci": wilson(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0),
        "specificity": specificity,
        "specificity_ci": wilson(tn, tn + fp) if (tn + fp) > 0 else (0.0, 0.0),
        "f1": f1,
        "f05": f05,
    }


CLASSES_3 = ["qualifies", "does_not_qualify", "insufficient_evidence"]


def compute_multiclass_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Compute 3-class confusion matrix, per-class precision/recall/F1 + Wilson 95% CIs,
    macro-F1, abstention rate, and abstention error rate (WS13-C).
    """
    n = len(y_true)
    cm: Dict[str, Dict[str, int]] = {t: {p: 0 for p in CLASSES_3} for t in CLASSES_3}
    for t, p in zip(y_true, y_pred):
        t_key = t if t in cm else "does_not_qualify"
        p_key = p if p in cm[t_key] else "does_not_qualify"
        cm[t_key][p_key] += 1

    per_class: Dict[str, Any] = {}
    f1_vals: List[float] = []
    for cls in CLASSES_3:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in CLASSES_3 if other != cls)
        fn = sum(cm[cls][other] for other in CLASSES_3 if other != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_vals.append(f1)
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "precision_ci": wilson(tp, tp + fp),
            "recall": rec,
            "recall_ci": wilson(tp, tp + fn),
            "f1": f1,
        }

    # Macro-F1 computed over all three classes regardless of presence in sample (§0.5 contract)
    macro_f1 = sum(per_class[c]["f1"] for c in CLASSES_3) / len(CLASSES_3)

    n_abstain_pred = sum(1 for p in y_pred if p == "insufficient_evidence")
    abstention_rate = n_abstain_pred / n if n > 0 else 0.0

    # Abstention error: predicting does_not_qualify when gold is insufficient_evidence
    n_gold_insuff = sum(1 for t in y_true if t == "insufficient_evidence")
    n_abst_err = sum(1 for t, p in zip(y_true, y_pred) if t == "insufficient_evidence" and p == "does_not_qualify")
    abstention_error_rate = n_abst_err / n_gold_insuff if n_gold_insuff > 0 else 0.0

    exact_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n > 0 else 0.0
    exact_hits = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    # Qualifier binary slice for direct FP/FN tracking
    q_tp = cm["qualifies"]["qualifies"]
    q_fp = cm["does_not_qualify"]["qualifies"] + cm["insufficient_evidence"]["qualifies"]
    q_fn = cm["qualifies"]["does_not_qualify"] + cm["qualifies"]["insufficient_evidence"]

    return {
        "n": n,
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "accuracy": exact_acc,
        "accuracy_ci": wilson(exact_hits, n),
        "abstention_rate": abstention_rate,
        "abstention_count": n_abstain_pred,
        "abstention_error_rate": abstention_error_rate,
        "abstention_error_ci": wilson(n_abst_err, n_gold_insuff) if n_gold_insuff > 0 else (0.0, 0.0),
        "qualifies_tp": q_tp,
        "qualifies_fp": q_fp,
        "qualifies_fn": q_fn,
    }


def compute_ece(probs: List[float], corrects: List[int], n_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute 10-bin Expected Calibration Error (ECE) and reliability diagram bin stats."""
    if not probs or len(probs) != len(corrects):
        return 0.0, []
    n = len(probs)
    ece = 0.0
    bins: List[Dict[str, Any]] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idxs = [
            i for i, p in enumerate(probs)
            if (p >= lo and (p < hi or (b == n_bins - 1 and p <= hi)))
        ]
        if not idxs:
            bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0, "mean_conf": None, "empirical_acc": None})
            continue
        mean_conf = sum(probs[i] for i in idxs) / len(idxs)
        emp_acc = sum(corrects[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(emp_acc - mean_conf)
        bins.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": len(idxs),
            "mean_conf": round(mean_conf, 4),
            "empirical_acc": round(emp_acc, 4),
        })
    return round(ece, 4), bins

