# implementation.md — v3.2 (Final) Implementation Plan

## Criterion-Aware Multi-Condition Classification, Three-Class Decisions, Evidence Validation, Real Calibration, and Reviewer-Critique Reconciliations

**Candidate:** Ruhaan Kakar  
**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**Task:** Mini Literature Classification Engine — Final Canonical Spec (supersedes v3.0 and v3.1 drafts)  
**Status:** Final. This document is the single source of truth. WS1–WS11 stand as amended by WS12–WS16 and reconciled by WS17 below.  
**Version target:** `3.2.0` (frozen tag: `v3.2-frozen`)

---

## 0. Canonical Spec Table (normative — every other section defers to this)

This table is the contract the code must match. CI enforces it via `tests/unit/test_doc_drift.py` (WS17-R1).

| Spec | Canonical value |
| --- | --- |
| Combiner rule count | **12** — `R1 no_text`, `R2 review_veto`, `R3a reagent_veto`, `R3b conflicting_evidence`, `R4 entity_absent`, `R5 background_citation`, `R6 wrong_system`, `R7 system_unstated`, `R8 different_activity`, `R9 unstated_threshold`, `R10 unsupported_criterion_clause`, `R11 all_satisfied` |
| Reason registry size | **18 codes** (17 v3 codes + `unstated_threshold` as #18; `system_mismatch_abstract` retained as deprecated, never emitted — 17 reachable active codes) |
| `mention_only` / `background_citation` class | `does_not_qualify`, `kind = "evidence-based"` (registry is authoritative; see WS17-R2) |
| `entity_role` arity | **2-way**: `product` or `agent` |
| Parser strictness | **Hard HTTP 422** on unparsed clauses; `allow_partial_criterion=True` is the only escape hatch |
| Calibration scheme | **LOEO on the fixed dev entity set read from `eval/splits.json`** |
| Calibration maps | Two MLE-fitted Platt maps: qualifier path `(A, B)` and negative path `(A_neg, B_neg)` fitted **only** on `different_activity` and `wrong_system` rows |
| Confidence emission | `qualifies` → calibrated `P(correct)`; `does_not_qualify` via `different_activity`/`wrong_system` → calibrated negative confidence; all other non-qualifiers (`mention_only`, `background_citation`, vetoes, all abstentions) → `confidence: null` |
| Dev entities | `["vanillin", "caffeine", "curcumin", "chitin"]` (per `eval/splits.json`) |
| Test entities | `["palladium(II) acetate", "artemisinic acid", "lipase"]` (per `eval/splits.json`) |
| Noisy real-world stratum | 10–15 audited real Europe PMC abstracts per test entity (~30–45 total), 20–30% double-labeled with Cohen's κ |

---

## 1. Round-3 Reconciliations (WS17)

Execution order: **R1 → R2 → R3 → R5 → R4 → R6**.

### WS17-R1 — One Spec, Doc ≡ Code
1. This file's body (WS1–WS17) is the sole authoritative plan.
2. §0 fixes every contested parameter (`len(RULES) == 12`, `len(REASON_REGISTRY) == 18`, 2-way `entity_role`, hard HTTP 422, LOEO on `eval/splits.json`, two-map calibration).
3. **Doc-drift CI guard** (`tests/unit/test_doc_drift.py`): parses `implementation.md`, verifies `len(RULES) == 12`, `len(REASON_REGISTRY) == 18`, 17 reachable combiner codes, `v2-baseline` tag pointing to `85894a0614dfc16f82bce76609a49c6b164b368e`, and asserts no contradictory trailing summary section exists.

### WS17-R2 — Canonical Reason Registry (18 Registered Codes, 17 Active Reachable)
1. `mention_only` and `background_citation` are updated to `label="does_not_qualify"`, `kind="evidence-based"` in `app/reasons.py`.
2. `unstated_threshold` (WS14-D numeric bounds) is registered as **code #18**, `label="insufficient_evidence"`, `kind="abstain"`.
3. `system_mismatch_abstract` remains in `REASON_REGISTRY` marked deprecated (never emitted under D6).
4. Complete 18-code registry:
   - **Veto (2):** `review_article`, `reagent_role`
   - **Evidence-based negatives (5):** `different_activity`, `wrong_system`, `entity_absent`, `background_citation`, `mention_only`
   - **Abstain (11 registered, 10 active reachable):** `no_text`, `abstract_only`, `activity_unclear`, `entity_unclear`, `system_unstated`, `system_mismatch_abstract` *(deprecated)*, `conflicting_evidence`, `evidence_split`, `unsupported_criterion_clause`, `possible_review`, `unstated_threshold`

### WS17-R3 — Zero Train/Test Leakage (`eval/splits.json`)
1. `eval/splits.json` is the single definition of dev and test entities:
   - `dev_entities`: `["vanillin", "caffeine", "curcumin", "chitin"]`
   - `test_entities`: `["palladium(II) acetate", "artemisinic acid", "lipase"]`
2. All threshold tuning (`TAU`, `DELTA`, `P_THIS_WORK_MIN`) and Platt MLE fitting (`(A, B)`, `(A_neg, B_neg)`, `NEG_FIT_WINDOW`) occur on dev LOEO folds only.
3. Leakage guards: `set(dev_entities) & set(test_entities) == set()`, `eval/calibrate.py` hard-fails on any test-entity row, and `config/calibration.json` stores `splits_sha256` audited under strict evaluation.

### WS17-R5 — Combiner Ordering & Bracket-Depth Clause Splitting
1. **12-rule combiner (`app/combiner.py` `RULES`):**
   - `R1`: `no_text` → `insufficient_evidence(no_text)`
   - `R2`: `review_veto` (`pubType == review` and `scope == this_study`) → `does_not_qualify(review_article)` **[VETO]**
   - `R3a`: `reagent_veto` (`entity_role == product`, no conflicting affirmative synthesis sentence) → `does_not_qualify(reagent_role)` **[VETO]**
   - `R3b`: `conflicting_evidence` (`entity_role == product`, role veto + affirmative synthesis claim elsewhere) → `insufficient_evidence(conflicting_evidence)`
   - `R4`: `entity_absent` (full text absent) / `entity_unclear` (abstract-only absent)
   - `R5`: `background_citation` (activity supported only in prior-art clauses) → `does_not_qualify(background_citation)`
   - `R6`: `wrong_system` (full-text or abstract-explicit host contradiction, D6) → `does_not_qualify(wrong_system)`
   - `R7`: `system_unstated` (host required, silent in abstract) → `insufficient_evidence(system_unstated)`
   - `R8`: `different_activity` (alternative activity / excluded method / mention-only) → `does_not_qualify(...)`
   - `R9`: `unstated_threshold` (numeric bound unverifiable from text) → `insufficient_evidence(unstated_threshold)`
   - `R10`: `unsupported_criterion_clause` (`unparsed_clauses` present under `allow_partial_criterion=True`) → `insufficient_evidence(unsupported_criterion_clause)`
   - `R11`: `all_satisfied` → `qualifies(reason_code=None)`
2. **Bracket-depth clause scanner (`app/text_utils.py`):** Tracks nesting depth across `([{` and `)]}`. Splits sentences ≥ `CLAUSE_SPLIT_MIN = 180` characters strictly at depth 0 (`;` → `, however,` / `, whereas,` / `, although,` / `, but ` → `, ` fallback), keeping IUPAC names like `2,2'-bipyridine`, `N,N-dimethyltryptamine`, and `dibenzo[b,d]oxepine` intact.

### WS17-R4 — Negative Platt Map (`different_activity`, `wrong_system` only; D5)
1. Negative Platt map `(A_neg, B_neg)` is fitted via logistic MLE **only** on `different_activity` and `wrong_system` dev rows using negative margin `m_neg`.
2. Pure topical `mention_only`, `background_citation`, vetoes (`review_article`, `reagent_role`), and all `insufficient_evidence` paths emit `confidence: null` (enforced via Pydantic validator on `PaperClassificationResult`).
3. Regression fixture: *"Caffeine intake was recorded by questionnaire"* yields `null`/low-confidence non-qualifier, never a false high confidence.

### WS17-R6 — Scoped Noisy Real-World Stratum
- `eval/labels.csv` includes `gold` (`qualifies | does_not_qualify | insufficient_evidence`) and `stratum` (`clean | noisy`), adding **10–15 audited real Europe PMC abstracts per test entity** (~30–45 total), with 20–30% double-labeled and Cohen's κ reported.

---

## 2. Core Workstreams Summary (WS1–WS16)

- **WS1 / WS14-D (`app/criterion_parser.py`):** Typed `CriterionSpec` with `entity_role ∈ {"product", "agent"}`, `in_vitro` system family, and G3 `ClauseConstraint` modifiers (`exclusion`, `numeric_threshold` with unit normalization across `g/L`, `mg/L`, `%`, `mol%`, compound `and`/`;` clauses). Unparsed clauses return HTTP 422 unless `allow_partial_criterion=True`.
- **WS2 (`app/reasons.py`):** Closed 18-code registry enforced on every payload.
- **WS3 / WS14-A,B,C,E (`app/conditions.py`):**
  - `check_c1_entity`: Boundary lookarounds, related-compound collision guard, asymmetric `product`/`agent` role polarity, and `synthesis_claim_sents` detection for R3b conflict routing.
  - `check_c2_activity`: Clause-split NLI scoring (no 250-char truncation), clause-local `p_this_work`, anaphora antecedent premise pairing (`ANAPHORA_WINDOW = 2`), exclusion & numeric threshold checking.
  - `check_c3_system`: Clause-attached organism matching, comparative/concessive exclusion (`unlike X,`, `whereas X,`), `in_vitro` separation from `microbial`, and D6 explicit abstract contradiction → `wrong_system`.
- **WS4 / WS16-A (`app/combiner.py`):** 12 ordered pure rules (`RULES`) verified by exhaustive truth-table testing including `allow_partial_criterion` interactions.
- **WS5 / WS15-A,B,C (`app/text_utils.py`, `app/classifier.py`):** Structural title/abstract segregation (title list never enters candidate gating or evidence selection), bracket-aware clause splitting, and byte-exact slice invariant.
- **WS6 / WS12-A,B (`app/scoring.py`, `eval/calibrate.py`, `config/calibration.json`):** Real Newton-Raphson logistic MLE on dev LOEO folds (qualifier path with `NEG_FIT_WINDOW = 0.15`; negative path on `different_activity`/`wrong_system`), loading `config/calibration.json` dynamically (`platt-v3.2-mle`), failing loudly on placeholder constants.
- **WS7–WS9 / WS13 (`data/snapshots/`, `eval/run.py`, `eval/metrics.py`, `notebooks/evaluation.ipynb`):** Complete snapshot manifest (`manifest.json`) with SHA-256 drift audit, 3-class evaluation + `abstention_error_rate` + Wilson 95% CIs + exact 3-class McNemar, and clean-room reproducible `DebertaNLIScorer` execution in `notebooks/evaluation.ipynb`.

---

## 3. Constants Register Reference

Full register maintained in `config/constants.md`:
- `TAU = 0.45`, `DELTA = 0.00`, `NEG_FIT_WINDOW = 0.15`, `ANAPHORA_WINDOW = 2`, `CLAUSE_SPLIT_MIN = 180`, `P_THIS_WORK_MIN = 0.50`, `band_edges = 0.80 / 0.50`, `ambiguity_margin = 0.10`, `splits_source = eval/splits.json`.

---

## 4. Residual Limitations

1. **Abstract paywall truncation:** Methods/yields absent from abstracts yield honest abstentions (`system_unstated`, `unstated_threshold`).
2. **Anaphora scope:** Pattern-based resolver covers high-confidence referents within `ANAPHORA_WINDOW = 2`; complex discourse chains remain out of scope.
3. **Punctuation-driven clause splitting:** Run-on sentences without depth-0 punctuation boundaries are scored whole.
4. **Noisy-stratum precision:** 10–15 real abstracts per test entity establish directional robustness with disclosed wide Wilson CIs.
5. **Narrow `m_neg` calibration coverage:** Negative Platt parameters apply strictly to `different_activity` and `wrong_system`; categorical gates emit `confidence: null`.
