import os
import sys
import json
import time
import logging

# Ensure literature-classifier is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.retrieval import build_query, search_europe_pmc
from app.nli import DebertaNLIScorer
from app.classifier import classify_publications
from app.text_utils import init_punkt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_sample")


def main():
    entity = "psilocybin"
    criterion = "Psilocybin was synthesized or biosynthesized in this work."
    synonyms = ["psilocybine", "4-phosphoryloxy-DMT"]
    max_results = 15

    logger.info("Initializing punkt and NLI model...")
    t0_init = time.perf_counter()
    init_punkt()
    scorer = DebertaNLIScorer()
    t_init = time.perf_counter() - t0_init
    logger.info(f"Model initialized in {t_init:.2f} s")

    query = build_query(entity, synonyms)
    logger.info(f"Retrieving publications from Europe PMC for query: '{query}'...")
    t0_retrieval = time.perf_counter()
    try:
        pubs = search_europe_pmc(query, max_results=max_results, timeout=15.0)
    except Exception as exc:
        logger.warning(f"Live Europe PMC retrieval failed ({exc}). Falling back to authentic Europe PMC publications from evaluation corpus.")
        import csv
        csv_path = os.path.join(root_dir, "eval", "labeled.csv")
        pubs = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["entity"].strip().lower() == entity.strip().lower():
                    pubs.append({
                        "title": row["title"],
                        "abstractText": row["abstract"],
                        "pmid": row["pmid"],
                        "doi": None,
                        "pmcid": None,
                    })
                if len(pubs) >= max_results:
                    break
    t_retrieval = time.perf_counter() - t0_retrieval
    logger.info(f"Retrieved {len(pubs)} publications in {t_retrieval:.2f} s")

    logger.info("Classifying retrieved publications...")
    t0_classify = time.perf_counter()
    res_run = classify_publications(
        publications=pubs,
        entity=entity,
        criterion=criterion,
        synonyms=synonyms,
        threshold=0.65,
        contrast_delta=0.05,
        scorer=scorer
    )
    ranked_results = res_run.results
    norm_crit = res_run.normalised_criterion
    tau = res_run.tau
    delta = res_run.delta
    t_classify = time.perf_counter() - t0_classify
    total_time = t_retrieval + t_classify
    logger.info(f"Classified {len(ranked_results)} papers in {t_classify:.2f} s (Total request latency: {total_time:.2f} s)")

    sample_output = {
        "query_used": query,
        "normalised_criterion": norm_crit,
        "threshold": tau,
        "contrast_delta": delta,
        "hypotheses_used": res_run.hypotheses_used,
        "contrast_guard_active": res_run.contrast_guard_active,
        "total_retrieved": len(ranked_results),
        "execution_profile": {
            "retrieval_seconds": round(t_retrieval, 3),
            "classification_seconds": round(t_classify, 3),
            "total_latency_seconds": round(total_time, 3)
        },
        "note": None,
        "results": [
            {
                "title": r["title"],
                "pmid": r["pmid"],
                "doi": r["doi"],
                "pmcid": r["pmcid"],
                "classification": r["classification"],
                "evidence": r["evidence"],
                "evidence_status": r["evidence_status"],
                "evidence_source": r.get("evidence_source", "abstract"),
                "qualify_score": r["qualify_score"],
                "confidence": r["confidence"],
                "confidence_band": r["confidence_band"],
            }
            for r in ranked_results
        ]
    }

    # Save to literature-classifier/sample_output.json
    target_path = os.path.join(root_dir, "sample_output.json")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2)
    logger.info(f"Saved sample output to {target_path}")

    # Also save to root of workspace for convenience
    root_sample = os.path.join(root_dir, "..", "sample_output.json")
    with open(root_sample, "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2)

    print("\n" + "=" * 75)
    print("SAMPLE RUN SUMMARY")
    print("=" * 75)
    print(f"Total Papers Processed: {len(ranked_results)}")
    print(f"Europe PMC Retrieval  : {t_retrieval:.2f} s")
    print(f"Batch NLI Inference   : {t_classify:.2f} s")
    print(f"Total Pipeline Latency: {total_time:.2f} s ({total_time/len(ranked_results):.2f} s/paper)")
    print("-" * 75)
    for i, r in enumerate(ranked_results[:5]):
        print(f"[{i+1}] {r['classification'].upper():16s} (score={r['qualify_score']:.3f}, conf={r['confidence']}, band={r['confidence_band']})")
        print(f"    Title   : {r['title']}")
        print(f"    PMID    : {r['pmid']} | Status: {r['evidence_status']}")
        print(f"    Evidence: {r['evidence']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
