import os
import sys
import csv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.nli import DebertaNLIScorer
from app.text_utils import candidate_sentences, build_entity_regex

scorer = DebertaNLIScorer()

with open(os.path.join(root_dir, "eval", "labeled.csv"), "r", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["entity"] == "vanillin"]

entity = "vanillin"
entity_regex = build_entity_regex(entity, ["4-hydroxy-3-methoxybenzaldehyde"])

hyps = {
    "P1_passive": "Vanillin is synthesised, biosynthesised, or produced in this work",
    "P2_agentive": "The authors synthesized, biosynthesized, or produced vanillin in this study.",
    "P3_product": "Vanillin was obtained as a synthesized or biosynthesized product."
}

print("MARGIN ANALYSIS PER PAPER:")
for r in rows:
    gold = r["gold_label"].strip()
    cands, _, _ = candidate_sentences(r["title"], r["abstract"], entity_regex, max_candidates=10)
    print(f"\nPMID {r['pmid']} | Gold: {gold}")
    if not cands:
        print("  NO CANDIDATES")
        continue
    for name, h in hyps.items():
        # Score candidates against h
        pairs = [(c, h) for c in cands]
        scores = scorer.score(pairs)
        margins = [s[0] - s[2] for s in scores]
        best_m = max(margins)
        best_c = cands[margins.index(best_m)]
        print(f"  {name:12s}: best_m = {best_m:+.3f} | '{best_c[:60]}...'")

