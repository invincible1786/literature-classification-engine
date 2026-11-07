"""Command Line Interface for offline reproducible execution and evaluation across v2 and v3 pipelines."""
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import ClassifyRequest
from app.classifier import classify_publications, DEFAULT_THRESHOLD, DEFAULT_CONTRAST_DELTA
from app.retrieval import load_snapshot_papers, search_europe_pmc, search_europe_pmc_dual, deduplicate_papers
from app.nli import DebertaNLIScorer, StubScorer
from app.text_utils import init_punkt
from app.hypotheses import detect_intent


def main():
    parser = argparse.ArgumentParser(description="Mini Literature Classification CLI")
    parser.add_argument("--entity", type=str, default="vanillin", help="Target entity")
    parser.add_argument("--criterion", type=str, default="The paper describes the synthesis or production of vanillin.", help="Inclusion criterion")
    parser.add_argument("--pipeline", type=str, choices=["v2", "v3"], default="v3", help="Pipeline version to run: 'v3' (default multi-condition 3-class) or 'v2' (legacy binary)")
    parser.add_argument("--offline", action="store_true", help="Load cached snapshots from data/snapshots/ without network requests")
    parser.add_argument("--stub", action="store_true", help="Use fast in-memory StubScorer instead of DeBERTa")
    parser.add_argument("--output", type=str, default="sample_output.json", help="Output file path")
    parser.add_argument("--max-results", type=int, default=10, help="Max publications to evaluate")
    parser.add_argument("--tau", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold tau")
    parser.add_argument("--delta", type=float, default=DEFAULT_CONTRAST_DELTA, help="Contrast guard delta")
    parser.add_argument("--abstract-negative-policy", type=str, choices=["conservative", "strict"], default="conservative", help="Policy for abstract-only system mismatch")
    parser.add_argument("--allow-partial-criterion", action="store_true", help="Allow evaluating supported core if unparsed clauses exist")
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
        dedup_count = 0
    else:
        intent = detect_intent(args.criterion, args.entity)
        query_intent = f'TITLE_ABS:(("{args.entity}") AND (synthesis OR biosynthesis OR "produced by" OR fermentation OR preparation OR yielded))'
        query_plain = f'TITLE_ABS:"{args.entity}"'
        print(f"Querying Europe PMC (pipeline={args.pipeline})...")
        if args.pipeline == "v3":
            publications, dedup_count = search_europe_pmc_dual(
                query_intent=query_intent,
                query_plain=query_plain,
                max_results=args.max_results,
                offline_fallback_entity=args.entity
            )
        else:
            raw = search_europe_pmc(query_intent, max_results=args.max_results, offline_fallback_entity=args.entity)
            publications, dedup_count = deduplicate_papers(raw)

    print(f"Classifying {len(publications)} publications with pipeline={args.pipeline}...")
    t0 = time.perf_counter()
    res = classify_publications(
        publications=publications,
        entity=args.entity,
        criterion=args.criterion,
        tau=args.tau,
        delta=args.delta,
        scorer=scorer,
        pipeline=args.pipeline,
        abstract_negative_policy=args.abstract_negative_policy,
        allow_partial_criterion=args.allow_partial_criterion
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    output_payload = {
        "api_version": args.pipeline,
        "query_entity": args.entity,
        "criterion": args.criterion,
        "total_retrieved": len(publications),
        "total_classified": len(res.results),
        "execution_time_ms": round(elapsed_ms, 2),
        "model_revision": getattr(scorer, "revision", "v3"),
        "ensemble_size": len(res.hypotheses_used),
        "hypotheses_used": res.hypotheses_used,
        "contrast_guard_active": res.contrast_guard_active,
        "guard_arming": res.guard_arming,
        "results": res.results
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, default=str)

    print(f"Successfully wrote {len(res.results)} classified papers to {args.output} in {elapsed_ms:.1f}ms")


if __name__ == "__main__":
    main()
