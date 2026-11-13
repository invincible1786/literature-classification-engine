# Project Handoff & Decision Context (`CLAUDE_CONTEXT.md`)

**Generated:** 2026-11-12  
**Workspace:** `c:\Users\Ruhaan Kakar\Desktop\loopworm` (active engine inside `literature-classifier/`)  
**Git HEAD:** `d3da1894dbe8ad30448db11137879980bd1d9987` (`v2-baseline` tag pinned at `85894a0614dfc16f82bce76609a49c6b164b368e`)  
**Working Tree Status:** All v3.2 code, snapshots, notebooks, reports, docs, and `eval_runs/` outputs are currently **uncommitted**; `v3.2-frozen` has **not** been tagged yet.

---

## 1. Non-Negotiable Constraints
1. **Zero paid LLM APIs:** Local inference only using `cross-encoder/nli-deberta-v3-small` (pinned rev `fa2804872c3b4bd748f38c0185cc85775361e735`, CUDA available).
2. **Extend existing app:** Keep `--pipeline v2` runnable unchanged.
3. **Not threshold tuning:** Accuracy/precision improvements must come from condition decomposition, role/conflict guards, and attribution/evidence rules rather than moving `tau`.
4. **Zero hallucinated evidence & strict title isolation:** Verbatim substring slice invariant (`canon_body[start:end] == display_text`); titles never returned as evidence.
5. **Reproducibility:** No `StubScorer` in `notebooks/evaluation.ipynb`; keep train/dev (`eval/splits.json`) disjoint from held-out test; never fudge labels or suppress invariant failures.

---

## 2. What Is Already Implemented & Verified (`v3.2`)

| Component | Status | Exact Files |
|---|---|---|
| **Architecture & Combiner** | Multi-condition (`C1` entity/role, `C2` NLI activity + `p_this_work >= 0.50`, `C3` host gazetteer + `in_vitro` + **D6** strict abstract `wrong_system`); **12** rules (`RULES`, `R1`–`R11` + `R3a`/`R3b` `conflicting_evidence`). | `app/conditions.py`, `app/combiner.py`, `config/systems.json`, `config/role_patterns.json` |
| **Criterion Grammar** | **G1** (product), **G2** (agent), **G3** (exclusions, unit-normalized thresholds, compounds); hard HTTP 422 unless `--allow-partial-criterion`. | `app/criterion_parser.py` |
| **Taxonomy & Calibration** | **18** registered reason codes (`17` reachable + deprecated `system_mismatch_abstract`); two L2-regularized Newton-Raphson MLE Platt curves (`config/calibration.json`: `platt-v3.2-mle`, dev entities `["vanillin", "caffeine", "curcumin", "chitin"]`); **D5** null confidence on abstentions, vetoes, `mention_only`, `background_citation`. | `app/reasons.py`, `app/scoring.py`, `app/schemas.py`, `eval/calibrate.py` |
| **Offline Benchmark (`N=56` Test)** | Real `DebertaNLIScorer` executed in `notebooks/evaluation.ipynb` over `152` SHA-256 audited records (`data/snapshots/manifest.json`). Test split (`palladium(II) acetate`, `artemisinic acid`, `lipase`; `20` clean + `36` noisy real Europe PMC): Qualifies Prec `0.6129` $\rightarrow$ **`0.9444`** (`12` FP $\rightarrow$ `1` FP), Rec `1.0000` $\rightarrow$ `0.8947`, Binary F1 `0.7600` $\rightarrow$ **`0.9189`** (McNemar `b=11, c=2, p=0.0225`), 3-class Macro-F1 `0.5245` $\rightarrow$ **`0.6381`** (Clean `0.9118`, Noisy `0.6230`), Abstention `0.1964`, ECE `0.3340` $\rightarrow$ **`0.0695`**. | `notebooks/evaluation.ipynb`, `eval/results_v2.json`, `eval/results_v3.json`, `eval/splits.json`, `eval/labels.csv` |
| **Offline Tests & Docs** | `pytest tests/unit tests/model -v`: **`62 passed`** (`50` unit + `12` real DeBERTa tests; Tier-3 live tests in `tests/live` not re-run). Docs synced at root and inside `literature-classifier/`. | `TECHNICAL_REPORT.md`, `ARCHITECTURE.md`, `README.md`, `KNOWN_ERRORS.md`, `Project.md`, `sample_output.json` |

---

## 3. Latest Live FastAPI Probe Findings (`eval_runs/`)

A read-only live run (`uvicorn app.main:app --port 8000`, `POST /classify` x 4 against live Europe PMC + PubChem) is saved in:
- `literature-classifier/eval_runs/run_20261112_220742.json`
- `literature-classifier/eval_runs/run_20261112_220742_meta.json`
- Checker: `literature-classifier/eval_runs/check_invariants.py`

### 3.1 Observed Invariant Violations (`40` Live Results Across `4` Queries)
Zero violations on **titles-as-evidence (`b=0`)**, **confidence rules (`d=0`)**, and **reason-registry mapping (`e=0`)**.  
Actual violations surfaced on live Europe PMC payloads:

1. **`c_insufficient_with_non_null_evidence` (5 occurrences):** `activity_unclear` (`evidence_status="insufficient"`) still attaches a non-null `evidence` candidate span (`42988866`, `36867394`, `42767251`, `au.174408492.20309692/v1`, `39705094`) instead of setting `evidence: null`.
2. **`a_mid_word_or_lowercase_cutoff` (6 occurrences) & `a_raw_vs_display_mismatch` (1 occurrence):**
   - `split_clauses_bracket_aware` extracts lowercase comma/sub-clauses as evidence strings (e.g., `"we herein demonstrated..."`, `"achieving a dihydroartemisinic acid titer of 79.4 mg/L."`, `"caffeine and diterpene content"`, `"EPS synthesis"`), rather than full grammatical sentences or properly bounded parent sentences.
   - `10.21203/rs.3.rs-3926665/v1` has multi-space formatting in `raw_text` (`"Abstract      Artemisinic..."`) that does not match collapsed whitespace in `display_text` under `clean_markup_and_unescape(raw_text) == display_text`.
3. **`f_related_compound_collision_on_qualifies` (1 occurrence — Query 2 `42107802`):** Because clause splitting selected only the final clause `"achieving a dihydroartemisinic acid titer of 79.4 mg/L."`, the returned qualifier evidence mentions only `dihydroartemisinic acid` (though `artemisinic acid` appeared earlier in the full abstract/sentence).
4. **`h_invalid_dedup_of` (1 occurrence — Query 1 `41474478`):** `dedup_of` was populated with literal `"unknown"` instead of `null` or a valid canonical `paper_id`.

### 3.2 Observed Live False Positives & Parser Gaps (Beyond Static Snapshot)
- **Query 1 (`Psilocybin`, `3` qualifies, `6` DNQ, `1` abstain):** Mostly strong; `40177917` (*E. coli* de novo synthesis) and `43014316` correctly qualified; 4 reviews cleanly rejected (`review_article`).
- **Query 2 (`Artemisinic Acid`, `6` qualifies — high FP rate on live query):**
  - **Parser gap:** `"The compound was produced by engineered microbial biosynthesis in this study."` parsed `activity='microbial biosynthesis'` and **`system=None`** (because `"microbial"` preceded `"biosynthesis"` instead of `"in <system>"`). Because `system` was `None`, `C3` did not reject plant (`Artemisia annua` in `rs-3926665/v1`) or oral caries (`42774055`) papers.
  - **Negation miss:** `40105756` matched *"without going through artemisinic acid (AA)"* as `qualifies`.
  - **Tiny fragment & live review leakage:** `42774055` qualified on 2-word fragment `"EPS synthesis"`; `38474639` qualified live when `"thereby significantly bolstering..."` wasn't filtered by review metadata/rules.
- **Query 3 (`Caffeine`, `6` qualifies — high FP rate on live query):**
  - **Antonym/degradation miss:** `43003031` (*"degrading caffeine"*) and `41275554` (*"caffeine degradation pathways"*) scored `qualifies` (`0.8931`, `0.8564`).
  - **Chemical vs. biological modifier ignored:** `"Caffeine was chemically synthesized in this work."` parsed `activity='synthesis or production'` without enforcing chemical route vs. plant gene regulation (`42563398`, `42970870`) or tea fermentation (`10.21203/rs.3.rs-11403610/v1`).
  - **Tiny fragment:** `42855060` qualified on `"caffeine and diterpene content"`.
- **Query 4 (`Palladium Acetate`, `1` qualifies, `6` DNQ, `3` abstain):** Strong overall (`reagent_role` and `mention_only` caught 6 catalysts); only 1 FP (`41173897`: *"synthesized through a hydrothermal reaction utilizing palladium acetate salt..."* — missing `"utilizing ... {E}"` in `config/role_patterns.json`).

---

## 4. Decision Options for Next Steps

### Option A — Freeze & Submit As-Is (Fastest, Zero Risk to Current Verified Suite)
- Everything requested by `todo.txt` and `implementation.md` v3.2 is already built, calibrated, notebook-executed, documented in `TECHNICAL_REPORT.md`, and passing `62/62` offline tests.
- You can optionally commit & tag `v3.2-frozen`, and frame Section 3 findings cleanly as **empirical live-query stress testing** demonstrating honest self-audit of residual failure modes.

### Option B — Targeted Surgical Fixes Before Freezing (Highest Live Quality Impact)
Fix only the high-ROI, zero-risk defects exposed by `check_invariants.py`, then re-run calibration, notebook, live queries, and tests:
1. **Strict `evidence: null` on `insufficient_evidence` (`c_` fix):** Null out top-level `evidence` / `evidence_sentence` / `evidence_offsets` when `evidence_status == "insufficient"` (keep diagnostic text inside `conditions.activity.evidence` only if needed, or preserve existing tests — check `test_v3_remediation.py` expectations first).
2. **Fix `dedup_of: "unknown"` (`h_` fix):** Inspect `app/retrieval.py` deduplication so missing survivor ID leaves `dedup_of = None` instead of `"unknown"`.
3. **Fix raw vs. display whitespace mapping (`a_` fix):** Preserve parent sentence context or ensure `clean_markup_and_unescape(raw_text)` matches `display_text`, and prevent bare noun fragments (`len < 30` or lacking finite verb, e.g. `"EPS synthesis"`, `"caffeine and diterpene content"`) from qualifying as standalone `C2` evidence (also avoids `42107802` qualifying on a subclause lacking parent compound `artemisinic acid`).
4. **Extract implicit `system` from `<system> biosynthesis` in `criterion_parser.py`:** e.g., `"engineered microbial biosynthesis"` sets `system.family = "microbial"` cleanly without requiring `"in microbial"`.
5. **Block obvious negation & degradation antonyms in synthesis intent + add `"utilizing {E}"`:**
   - Reject synthesis candidates containing explicit degradation antonyms (`degrading`, `degradation`, `decaffeination`) or `"without going through {E}"`.
   - Add `utilizing\s+(?:the\s+)?(?:[a-zA-Z0-9\-]+\s+){0,2}{E}` (or similar catalyst/precursor verb form) to `config/role_patterns.json`.

### Option C — Clean Up Workspace & Git History Only
- Decide whether to keep `eval_runs/` untracked (or add to `.gitignore` / keep for appendix), verify `tests/live`, commit the `v3.2` changes cleanly across logical commits, and tag `v3.2-frozen`.

