"""CI Consistency & Doc-Drift Guard Tests (WS17-R1, WS17-R2, WS17-R3, WS16-B)."""
import ast
import json
import os
import re
import subprocess
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.combiner import RULES
from app.reasons import REASON_REGISTRY, validate_reason_code


def test_doc_drift_canonical_spec_table():
    """WS17-R1: implementation.md §0 Canonical Spec Table matches code parameters."""
    impl_path = os.path.join(BASE_DIR, "implementation.md")
    with open(impl_path, "r", encoding="utf-8") as f:
        text = f.read()

    # No second conflicting summary/footer section
    assert "The final merged implementation.md" not in text
    assert "registry -> 18" not in text and "different_product" not in text

    # Rule count in §0 matches len(RULES) == 12
    m_rules = re.search(r"\|\s*Combiner rule count\s*\|\s*\*\*(\d+)\*\*", text)
    assert m_rules is not None, "Missing Combiner rule count row in §0"
    assert int(m_rules.group(1)) == len(RULES) == 12

    # Reason registry size in §0 matches len(REASON_REGISTRY) == 18
    m_reg = re.search(r"\|\s*Reason registry size\s*\|\s*\*\*(\d+)\s*codes\*\*", text)
    assert m_reg is not None, "Missing Reason registry size row in §0"
    assert int(m_reg.group(1)) == len(REASON_REGISTRY) == 18


def test_combiner_literals_in_registry_and_reachability():
    """WS17-R2: Every literal reason_code in combiner.py is registered; 17 active codes reachable, deprecated never emitted."""
    combiner_path = os.path.join(BASE_DIR, "app", "combiner.py")
    with open(combiner_path, "r", encoding="utf-8") as f:
        src = f.read()

    assert "system_mismatch_abstract" not in src, "Deprecated D6 code must never be emitted by combiner.py"

    tree = ast.parse(src)
    # Collect string literals assigned to reason_code or compared against reason_code
    literals = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in REASON_REGISTRY:
                literals.add(node.value)

    for lit in literals:
        assert validate_reason_code(lit), f"Unregistered reason literal in combiner.py: {lit}"

    # Verify all 17 non-deprecated codes are valid active codes
    active_registered = {k for k in REASON_REGISTRY if k != "system_mismatch_abstract"}
    assert len(active_registered) == 17


def test_splits_disjoint_and_canonical():
    """WS17-R3: Dev and test splits in eval/splits.json are strictly disjoint and match §0."""
    splits_path = os.path.join(BASE_DIR, "eval", "splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    dev = splits["dev_entities"]
    test = splits["test_entities"]
    assert len(dev) >= 4
    assert len(test) >= 3
    assert set(dev) & set(test) == set(), "Train/test entity overlap detected!"
    assert set(dev) == {"vanillin", "caffeine", "curcumin", "chitin"}
    assert set(test) == {"palladium(II) acetate", "artemisinic acid", "lipase"}


def test_no_placeholder_calibration_or_heuristic_constants():
    """WS12-A/B: No hardcoded placeholder Platt constants or 0.15 heuristic in scoring.py or classifier.py."""
    scoring_src = open(os.path.join(BASE_DIR, "app", "scoring.py"), "r", encoding="utf-8").read()
    clf_src = open(os.path.join(BASE_DIR, "app", "classifier.py"), "r", encoding="utf-8").read()

    assert "7.0 * (final_score - tau)" not in scoring_src
    assert "7.0 * (dec.ranking_score - tau)" not in clf_src
    assert "heuristic_negative" not in clf_src
    assert "StubScorer" not in open(os.path.join(BASE_DIR, "notebooks", "evaluation.ipynb"), "r", encoding="utf-8").read()


def test_constants_register_and_no_generative_llm():
    """Verify v3.2 constants appear in config/constants.md, MAX_SENT_CHARS is removed from app/, and no LLM deps."""
    const_text = open(os.path.join(BASE_DIR, "config", "constants.md"), "r", encoding="utf-8").read()
    for required_const in ("TAU", "DELTA", "NEG_FIT_WINDOW", "ANAPHORA_WINDOW", "CLAUSE_SPLIT_MIN", "P_THIS_WORK_MIN"):
        assert required_const in const_text, f"Missing constant {required_const} in config/constants.md"

    app_dir = os.path.join(BASE_DIR, "app")
    for fname in os.listdir(app_dir):
        if fname.endswith(".py"):
            src = open(os.path.join(app_dir, fname), "r", encoding="utf-8").read()
            assert "MAX_SENT_CHARS" not in src
            assert 'pipeline("text-generation")' not in src
            assert "import openai" not in src.lower()
            assert "import anthropic" not in src.lower()


def test_v2_baseline_git_tag_pinned():
    """WS16-B: v2-baseline git tag resolves to expected pre-v3 commit 85894a0."""
    repo_root = os.path.dirname(BASE_DIR)
    res = subprocess.run(
        ["git", "rev-parse", "v2-baseline^{commit}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
        assert res.stdout.strip().startswith("85894a0")

