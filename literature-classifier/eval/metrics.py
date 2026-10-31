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

