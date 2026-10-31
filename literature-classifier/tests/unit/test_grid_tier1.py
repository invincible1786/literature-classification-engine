"""Tier 1 Unit Test: E11 Criterion Arming Grid evaluation."""
import os
import csv
from app.hypotheses import lexical_intent_armed


def test_e11_criteria_grid_lexical_arming():
    """E11: Evaluate lexical guard arming across the ~20-criterion benchmark grid."""
    grid_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "eval",
        "criteria_grid.csv"
    )
    assert os.path.exists(grid_path), "eval/criteria_grid.csv must exist"

    results = []
    with open(grid_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["criterion_id"]
            group = row["group"]
            criterion = row["criterion"]
            expected = row["expected_arming"]

            # Assume entity = vanillin or curcumin
            entity = "vanillin" if "vanillin" in criterion else "curcumin"
            armed = lexical_intent_armed(criterion, entity=entity)

            results.append({
                "cid": cid,
                "group": group,
                "criterion": criterion,
                "expected": expected,
                "armed": armed
            })

    synthesis_rows = [r for r in results if r["group"] == "synthesis"]
    non_synth_rows = [r for r in results if r["group"] == "non_synthesis"]
    misspelled_rows = [r for r in results if r["group"] == "misspelled"]
    ambiguous_rows = [r for r in results if r["group"] == "ambiguous"]

    # 1. Synthesis criteria: 100% armed
    synth_armed_count = sum(1 for r in synthesis_rows if r["armed"])
    assert synth_armed_count == len(synthesis_rows), f"Synthesis arming failed: {synth_armed_count}/{len(synthesis_rows)}"

    # 2. Non-synthesis criteria: 0% armed (notably 'obtain ethical approval' must NOT arm)
    non_synth_armed_count = sum(1 for r in non_synth_rows if r["armed"])
    assert non_synth_armed_count == 0, f"Non-synthesis criteria falsely armed: {non_synth_armed_count}/{len(non_synth_rows)}"

    # 3. Misspelled criteria: 100% armed via restricted fuzzy match
    misspelled_armed_count = sum(1 for r in misspelled_rows if r["armed"])
    assert misspelled_armed_count == len(misspelled_rows), f"Misspelled criteria missed: {misspelled_armed_count}/{len(misspelled_rows)}"

    print(f"\nE11 Grid Summary:")
    print(f"  Synthesis:     {synth_armed_count}/{len(synthesis_rows)} armed")
    print(f"  Non-Synthesis: {len(non_synth_rows) - non_synth_armed_count}/{len(non_synth_rows)} correctly disarmed")
    print(f"  Misspelled:    {misspelled_armed_count}/{len(misspelled_rows)} armed (fuzzy)")
    print(f"  Ambiguous:     {sum(1 for r in ambiguous_rows if r['armed'])}/{len(ambiguous_rows)} armed")

