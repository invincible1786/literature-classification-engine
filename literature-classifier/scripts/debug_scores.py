import os
import sys

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.nli import DebertaNLIScorer

def debug_starting_material():
    scorer = DebertaNLIScorer()

    premise = "Compound 4 was prepared using psilocybin as a starting material."
    criterion = "Psilocybin was synthesized or biosynthesized in this work."

    contrast_hypotheses = [
        "Psilocybin is used as a reagent, starting material, or administered, not produced.",
        "Psilocybin is used as a starting material or reactant in this study.",
        "Psilocybin is used as a reagent, solvent, or standard, not produced.",
        "Psilocybin is administered to subjects, not synthesized.",
        "Psilocybin is a starting material or reactant in this reaction.",
    ]

    print("=" * 70)
    print("DEBUGGING SCORES FOR STARTING MATERIAL PREMISE")
    print(f"Premise  : {premise}")
    print(f"Criterion: {criterion}")
    print("-" * 70)

    # Score criterion
    crit_scores = scorer.score([(premise, criterion)])[0]
    p_ent, p_neu, p_con = crit_scores
    m = p_ent - p_con
    print(f"Criterion Result: P(ent)={p_ent:.4f}, P(neu)={p_neu:.4f}, P(con)={p_con:.4f} | Margin m={m:.4f}")

    print("\nContrast Hypotheses:")
    for h in contrast_hypotheses:
        scores = scorer.score([(premise, h)])[0]
        c_ent, c_neu, c_con = scores
        print(f"  P(ent)={c_ent:.4f}, P(con)={c_con:.4f} | H: {h}")

    print("=" * 70)

if __name__ == "__main__":
    debug_starting_material()

