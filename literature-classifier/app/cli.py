"""Command Line Interface for offline reproducible execution and evaluation."""
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import ClassifyRequest
from app.classifier import classify_publications, DEFAULT_THRESHOLD, DEFAULT_CONTRAST_DELTA
from app.retrieval import load_snapshot_papers, search_europe_pmc, build_query
from app.nli import DebertaNLIScorer, StubScorer
from app.text_utils import init_punkt


def main():
    parser = argparse.ArgumentParser(description="Mini Literature Classification CLI")
    parser.add_argument("--entity", type=str, default="vanillin", help="Target entity")
    parser.add_argument("--criterion", type=str, default="The paper describes the synthesis or production of vanillin.", help="Inclusion criterion")
    parser.add_argument("--offline", action="store_true", help="Load cached snapshots from data/snapshots/ without network requests")
    parser.add_argument("--stub", action="store_true", help="Use fast in-memory StubScorer instead of DeBERTa")
    parser.add_argument("--output", type=str, default="sample_output.json", help="Output file path")
    parser.add_argument("--max-results", type=int, default=10, help="Max publications to evaluate")
    parser.add_argument("--tau", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold tau")
    parser.add_argument("--delta", type=float, default=DEFAULT_CONTRAST_DELTA, help="Contrast guard delta")
    args = parser.parse_args()

    init_punkt()
    print(f"Loading scorer ({'StubScorer' if args.stub else 'DebertaNLIScorer'})...")
    scorer = StubScorer() if args.stub else DebertaNLIScorer()

    if args.offline:
        print(f"Loading offline snapshot for '{args.entity}'...")
        papers = load_snapshot_papers(args.entity)
        if not papers:
            print(f"Error: Snapshot for entity '{args.entity}' not found in data/snapshots/", file=sys.stderr)
            sys.exit(1)
        publications = papers[:args.max_results]
    else:
        query = build_query(args.entity)
        print(f"Querying Europe PMC for '{query}'...")
        publications = search_europe_pmc(query, max_results=args.max_results, offline_fallback_entity=args.entity)

    print(f"Classifying {len(publications)} publications...")
    t0 = time.perf_counter()
    res = classify_publications(
        publications=publications,
        entity=args.entity,
        criterion=args.criterion,
        threshold=args.tau,
        contrast_delta=args.delta,
        scorer=scorer
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    output_payload = {
        "query_entity": args.entity,
        "criterion": args.criterion,
        "total_retrieved": len(publications),
        "total_classified": len(res.results),
        "execution_time_ms": round(elapsed_ms, 2),
        "model_revision": getattr(scorer, "revision", "v1"),
        "ensemble_size": len(res.hypotheses_used),
        "hypotheses_used": res.hypotheses_used,
        "contrast_guard_active": res.contrast_guard_active,
        "guard_arming": res.guard_arming,
        "results": res.results
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Successfully wrote {len(res.results)} classified papers to {args.output} in {elapsed_ms:.1f}ms")


if __name__ == "__main__":
    main()

