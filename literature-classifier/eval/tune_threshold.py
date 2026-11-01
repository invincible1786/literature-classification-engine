import os
import sys
import csv
import logging
import statistics
from typing import List, Dict, Any, Tuple

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.nli import DebertaNLIScorer
from app.text_utils import candidate_sentences, build_entity_regex
from app.paraphrase import generate_criterion_paraphrases
from app.classifier import format_contrast_hypotheses

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tune_threshold")


def load_labeled_data(csv_path: str, entity_filter: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["entity"].strip().lower() == entity_filter.strip().lower():
                rows.append(r)
    return rows


def precompute_paper_scores(
    papers: List[Dict[str, Any]],
    scorer: DebertaNLIScorer
) -> List[Dict[str, Any]]:
    """
    Extracts candidate sentences and pre-computes criterion paraphrase and contrast NLI scores
    for each paper in a single batched pass.
    """
    entity = papers[0]["entity"]
    criterion = papers[0]["criterion"]
    criterion_paraphrases = generate_criterion_paraphrases(criterion, entity)
    contrast_hyps, _ = format_contrast_hypotheses(None, None, criterion, entity)
    entity_regex = build_entity_regex(entity)

    n_crit = len(criterion_paraphrases)
    n_cont = len(contrast_hyps)
    pairs_per_cand = n_crit + n_cont

    all_pairs: List[Tuple[str, str]] = []
    paper_spans: List[Tuple[int, int]] = []
    paper_candidates_list: List[List[str]] = []

    for p in papers:
        cands, _, _ = candidate_sentences(
            title=p.get("title"),
            abstract=p.get("abstract"),
            entity_regex=entity_regex,
            max_candidates=10
        )
        paper_candidates_list.append(cands)
        start = len(all_pairs)
        for c in cands:
            for cp in criterion_paraphrases:
                all_pairs.append((c, cp))
            for ch in contrast_hyps:
                all_pairs.append((c, ch))
        end = len(all_pairs)
        paper_spans.append((start, end))

    logger.info(f"Scoring {len(all_pairs)} total pairs across {len(papers)} papers using {n_crit} criterion paraphrases and {n_cont} contrast hypotheses...")
    all_scored = scorer.score(all_pairs) if all_pairs else []

    processed = []
    for idx, p in enumerate(papers):
        cands = paper_candidates_list[idx]
        start, end = paper_spans[idx]
        scores = all_scored[start:end]

        margins = []
        p_ents = []
        contrast_ents = []
        for c_i in range(len(cands)):
            offset = c_i * pairs_per_cand
            cand_crit_scores = scores[offset : offset + n_crit]
            cand_cont_scores = scores[offset + n_crit : offset + pairs_per_cand]

            cand_margins = [crit[0] - crit[2] for crit in cand_crit_scores]
            cand_p_ents = [crit[0] for crit in cand_crit_scores]

            m_i = statistics.median(cand_margins)
            p_ent_i = statistics.median(cand_p_ents)
            c_i = max((cont[0] for cont in cand_cont_scores), default=0.0)

            margins.append(m_i)
            p_ents.append(p_ent_i)
            contrast_ents.append(c_i)

        processed.append({
            "paper": p,
            "cands": cands,
            "margins": margins,
            "p_ents": p_ents,
            "contrast_ents": contrast_ents,
            "gold_label": p["gold_label"].strip()
        })
    return processed


def evaluate_threshold_grid(
    processed_papers: List[Dict[str, Any]],
    tau_grid: List[float],
    delta_grid: List[float]
):
    """
    Evaluates grid of (tau, delta) combinations on tuning set.
    """
    best_combo = None
    best_f1 = -1.0
    best_prec = -1.0
    records = []

    for tau in tau_grid:
        for delta in delta_grid:
            tp = fp = tn = fn = 0
            for item in processed_papers:
                cands = item["cands"]
                gold = item["gold_label"]

                if not cands or not item["margins"]:
                    pred = "does_not_qualify"
                else:
                    best_idx = max(range(len(item["margins"])), key=lambda k: item["margins"][k])
                    m_best = item["margins"][best_idx]
                    p_ent_best = item["p_ents"][best_idx]
                    c_best = item["contrast_ents"][best_idx]

                    contrast_fired = (c_best > p_ent_best + delta)
                    if m_best >= tau and not contrast_fired:
                        pred = "qualifies"
                    else:
                        pred = "does_not_qualify"

                if pred == "qualifies" and gold == "qualifies":
                    tp += 1
                elif pred == "qualifies" and gold == "does_not_qualify":
                    fp += 1
                elif pred == "does_not_qualify" and gold == "does_not_qualify":
                    tn += 1
                else:
                    fn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            records.append({
                "tau": tau, "delta": delta,
                "precision": prec, "recall": rec, "f1": f1,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            })

            # Optimize for high F1, with precision as tiebreaker
            score = (f1, prec)
            if score > (best_f1, best_prec):
                best_f1, best_prec = score
                best_combo = (tau, delta, prec, rec, f1, tp, fp, tn, fn)

    return records, best_combo


def main():
    csv_path = os.path.join(root_dir, "eval", "labeled.csv")
    entity_a = "psilocybin"

    papers_a = load_labeled_data(csv_path, entity_a)
    logger.info(f"Loaded {len(papers_a)} papers for Entity A ({entity_a}) [Tuning Set]")

    scorer = DebertaNLIScorer()
    processed_a = precompute_paper_scores(papers_a, scorer)

    tau_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    delta_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    records, best = evaluate_threshold_grid(processed_a, tau_values, delta_values)

    print("\n" + "=" * 65)
    print("THRESHOLD TUNING RESULTS (Entity A: Psilocybin)")
    print("=" * 65)
    print(f"{'tau':>6} | {'delta':>6} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'TP/FP/TN/FN':>14}")
    print("-" * 65)

    for r in records[::3]:
        print(f"{r['tau']:6.2f} | {r['delta']:6.2f} | {r['precision']:10.3f} | {r['recall']:8.3f} | {r['f1']:8.3f} | {r['tp']:2d}/{r['fp']:2d}/{r['tn']:2d}/{r['fn']:2d}")

    tau_star, delta_star, p_star, r_star, f1_star, tp, fp, tn, fn = best
    print("-" * 65)
    print(f"\nOPTIMAL TUNED PARAMETERS:")
    print(f"  Threshold tau*  = {tau_star:.2f}")
    print(f"  Contrast delta* = {delta_star:.2f}")
    print(f"  Tuning Metrics  : Precision = {p_star:.3f}, Recall = {r_star:.3f}, F1 = {f1_star:.3f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
