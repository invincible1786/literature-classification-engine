# Technical Report: Criterion-Aware Multi-Condition Biomedical Literature Classification (`v3.2`)

**Candidate:** Ruhaan Kakar  
**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**Project:** Mini Literature Classification Engine (`v3.2-frozen`, extending `v2-baseline` commit `85894a0`)  
**Environment & Model:** Python 3.11.0 | PyTorch 2.6.0+cu124 | `cross-encoder/nli-deberta-v3-small` (pinned rev `fa280487`) | Zero paid LLM APIs

---

## 1. Executive Challenge–Response Summary

An independent evaluation of `v2-baseline` raised five core architectural challenges: (1) multi-condition semantic verification rather than monolithic scoring, (2) interpretation of arbitrary user-defined criteria, (3) replacing coerced binary outputs with principled three-class abstention, (4) eliminating paraphrased or title-based evidence spans, and (5) genuine empirical evaluation with real out-of-fold Platt probability calibration rather than tuned thresholds or dummy placeholders.

| Reviewer Challenge | v3.2 Architectural Response | Primary Implementation |
|---|---|---|
| **Multi-condition semantic verification** | Decomposes classification into independent condition checks ($C_1$ Entity & role polarity, $C_2$ Activity & clause-local attribution, $C_3$ Host system) aggregated by an exhaustive 12-rule deterministic combiner (`RULES`, `R1`–`R11` with `R3a`/`R3b`). | [`app/conditions.py`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/app/conditions.py), [`app/combiner.py`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/app/combiner.py) |
| **User-defined criteria interpretation** | Parses arbitrary criterion text into `CriterionSpec` across grammar tiers **G1** (product), **G2** (agent), and **G3** (exclusions, unit-normalized numeric thresholds, compound clauses); unparsed clauses trigger hard HTTP 422 unless `--allow-partial-criterion` is explicitly set. | [`app/criterion_parser.py`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/app/criterion_parser.py) |
| **Three-class abstention & taxonomy** | Separates `qualifies`, `does_not_qualify`, and `insufficient_evidence` across an authoritative 18-code registry (`17` reachable active codes + deprecated `system_mismatch_abstract`); reagent-vs-synthesis conflicts route to `conflicting_evidence` (`R3b`) rather than blind veto. | [`app/reasons.py`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/app/reasons.py) |
| **Zero hallucinated evidence** | Structural title isolation (title tokenized separately from body and never passed into candidate or evidence selection) and runtime slice invariant `canon_body[start:end] == evidence.display_text`. | [`app/classifier.py`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/app/classifier.py) |
| **Real calibration & held-out audit** | Fits two regularized Newton-Raphson MLE Platt maps on the 4-entity dev LOEO split (`eval/splits.json`, zero overlap with 3 held-out test entities), evaluated over 152 SHA-256 audited records (`clean` + real Europe PMC `noisy` strata) with zero stub scorers. | [`eval/calibrate.py`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/eval/calibrate.py), [`notebooks/evaluation.ipynb`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/notebooks/evaluation.ipynb) |

---

## 2. Architecture & Methodology Delta (`v2-baseline` $\rightarrow$ `v3.2-frozen`)

The v2 pipeline remains runnable unchanged via `--pipeline v2`. The v3.2 upgrade replaces single-score ranking with five composable stages:

1. **Syntactic Criterion Parsing (`CriterionSpec`):** Parses user criteria into 2-way `entity_role` (`product | agent`), target `activity`, optional `system` (`microbial`, `yeast`, `bacteria`, `fungi`, `plant`, `mammalian`, `in_vitro`), `scope` (`this_study | any`), and `ClauseConstraint` modifiers (`G3`: exclusion phrases like *"not in yeast"*, numeric thresholds normalized across units such as `%`, `g/L`, `mg/L`, `mM`, and compound clauses).
2. **Bracket-Depth Clause Splitting & Anaphora (`app/text_utils.py`):** Replaces naive character truncation (`MAX_SENT_CHARS`, removed) with a parenthesis/bracket depth-0 clause splitter (`CLAUSE_SPLIT_MIN = 180`), preserving IUPAC chemical names (`2,2'-bipyridine`, `dibenzo[b,d]oxepine`), paired with backward chemical anaphora resolution (`ANAPHORA_WINDOW = 2`) for subjectless synthesis sentences (*"It was obtained in 88% yield"*).
3. **Clause-Local Multi-Condition Evaluation:**
   - **$C_1$ Entity:** Word-boundary lookaround regex over vetted aliases (`len >= 5`), related-compound guard (`artemisinin` vs `artemisinic acid`), and expanded prepositional reagent role detection (`config/role_patterns.json`) that permits up to two modifier tokens (e.g., *"mediated by the immobilized lipase"*). Crucially, if one clause flags reagent role while another clause in the abstract contains a synthesis claim (`synthesis_claim_sents`), rule **R3b** emits `insufficient_evidence(conflicting_evidence)` for curator review instead of blindly vetoing.
   - **$C_2$ Activity:** Evaluates positive activity paraphrases vs. negative role hypotheses and non-production contrast templates via `cross-encoder/nli-deberta-v3-small`, enforcing clause-local experimental attribution ($p_{\text{this\_work}} \ge 0.50$) and checking `G3` constraints and numeric thresholds (`unstated_threshold` when missing).
   - **$C_3$ System:** Evaluates comparative clauses (preventing *"Unlike E. coli systems, yeast produced..."* from passing `E. coli`) and separates `in_vitro` cell-free cascades from `microbial` hosts. Under **D6**, explicit contradictory host organisms in abstracts resolve directly to `does_not_qualify(wrong_system)`, while missing host mentions resolve to `insufficient_evidence(system_unstated)`.
4. **Two-Map Platt Calibration & D5 Null Policy (`app/scoring.py`, `eval/calibrate.py`):** Removes `v2` heuristic linear scaling `7.0 * (score - tau)`. Qualifier logits fit `(A = 3.459395, B = -1.162682)` on `n_total = 33` dev rows (`28` positives, `5` borderline negatives within `NEG_FIT_WINDOW = 0.15`, out-of-fold Brier = `0.1195`, LOEO `mean_A = 3.4353`, `std_A = 0.8332`, `slope_unstable_flag = false`). A second negative map `(A_neg = 1.549835, B_neg = -0.355466, n = 4, out_of_fold_brier = 0.1897)` calibrates **only** `different_activity` and `wrong_system`; all vetoes, `mention_only`, `background_citation`, and abstentions strictly emit `confidence: null` (**D5**).

---

## 3. Experimental Evaluation: Original (`v2`) vs. Revised (`v3.2`)

All numbers below come directly from the offline executed notebook [`notebooks/evaluation.ipynb`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/notebooks/evaluation.ipynb) running `cross-encoder/nli-deberta-v3-small` against SHA-256 audited snapshots (`manifest.json` `v3.2`, `152` records). Train/dev (`vanillin`, `caffeine`, `curcumin`, `chitin`; `splits_sha256: 2e7d2ea5...`) and held-out test (`palladium(II) acetate`, `artemisinic acid`, `lipase`; `N = 56`, `20` clean + `36` real Europe PMC noisy abstracts) are strictly disjoint. Inter-annotator agreement across `11` double-labeled noisy papers reached `0.9091` exact agreement (`cohens_kappa = 1.0`, excluding unresolved tie rows per `labels.csv` protocol; `17` double-labeled total at `0.9412`).

### 3.1 Held-Out Test Performance (`N = 56`: `clean = 20`, `noisy = 36`)

| Metric | `v2-baseline` | `v3.2-frozen` | Delta / Notes |
|---|---|---|---|
| **Qualifies Precision** | `0.6129` (`95% CI [0.4382, 0.7627]`) | **`0.9444`** (`95% CI [0.7424, 0.9901]`) | **`+0.3315`** (`12` FP $\rightarrow$ `1` FP) |
| **Qualifies Recall** | **`1.0000`** (`95% CI [0.8318, 1.0000]`) | `0.8947` (`95% CI [0.6861, 0.9706]`) | `-0.1053` (`2` noisy FN abstained/conflicted) |
| **Binary Qualifies F1 / F0.5** | `0.7600` / `0.6643` | **`0.9189` / `0.9341`** | **`+0.1589` F1 / `+0.2698` F0.5** |
| **Binary Qualifies Accuracy** | `0.7857` (`[0.6618, 0.8729]`) | **`0.9464`** (`[0.8539, 0.9816]`) | McNemar exact `b=11, c=2, p=0.0225` |
| **3-Class Macro-F1** | `0.5245` | **`0.6381`** | `+0.1136` (McNemar 3-class `b=6, c=5, p=1.00`) |
| **3-Class Accuracy** | `0.7679` | **`0.7857`** (`[0.6618, 0.8729]`) | Abstention rate: `0.0000` $\rightarrow$ `0.1964` (`11/56`) |
| **Abstention Error Rate** | `0.3333` (`3/3` gold abstain missed) | `0.3333` (`1/3` caught, `2` missed) | Thin `n=3` CI `[0.0615, 0.7923]` |
| **ECE (Reliability Calibration)** | `0.3340` (`ece_raw_baseline`) | **`0.0695`** (`ece_calibrated`) | **`-0.2645` (-79.2% calibration error)** |

### 3.2 Stratified Clean vs. Noisy Real Europe PMC Breakdown

| Stratum (`N`) | `v2` Macro-F1 | `v3.2` Macro-F1 | `v2` Qualifies FP | `v3.2` Qualifies FP | `v3.2` Acc (`95%` Wilson CI) | `v3.2` Abstain Rate |
|---|---|---|---|---|---|---|
| **Clean Synthetic/Curated (`20`)** | `0.8465` | **`0.9118`** | `3` | **`0`** | `0.8500` `[0.6396, 0.9476]` | `0.1500` (`3/20`) |
| **Noisy Real Europe PMC (`36`)** | `0.4921` | **`0.6230`** | `9` | **`1`** | `0.7500` `[0.5893, 0.8625]` | `0.2222` (`8/36`) |

*Threshold Sensitivity Check ($\pm 20\%$ around nominal $\tau = 0.45$ on held-out test):* `tau=0.36` gives `macro_f1=0.6474`, `qualifies_fp=2`, `abstain_rate=0.1607`; `tau=0.45` gives `macro_f1=0.6381`, `qualifies_fp=1`, `abstain_rate=0.1964`; `tau=0.54` gives `macro_f1=0.6249`, `qualifies_fp=1`, `abstain_rate=0.2143`. Precision gains stem from multi-condition rules and role vetoes rather than threshold tuning.

---

## 4. Representative Case Studies (Held-Out Real Europe PMC Abstracts)

1. **CS1 — Immobilized Catalyst Modifier (`lipase`, DOI `10.21203/rs.3.rs-9942287/v1`, noisy):** *v2* qualified the paper because it discussed *lipase* and ester polycondensation. *v3.2* matched the modifier-tolerant role pattern `"mediated by the immobilized lipase"` (`config/role_patterns.json`) in span `[335:443]` and rejected it deterministically as `does_not_qualify(reagent_role, confidence=null)`.
2. **CS2 — Epistemological / Unstated Review (`lipase` `42778850` & `artemisinic acid` `38474639`, noisy):** Both papers discussed engineered yeast production in abstracts and qualified under *v2*. *v3.2* detected review metadata/language via `R2`, returning `does_not_qualify(review_article, confidence=null, evidence=null)`.
3. **CS3 — Wrong Host System in Abstract (`artemisinic acid`, DOI `10.21203/rs.3.rs-3926665/v1` & PMID `42774055`, noisy):** Under microbial biosynthesis criteria, *v2* qualified both. Under D6 (`R6`), *v3.2* identified explicit native plant (*Artemisia annua*) and *in vivo* mammalian/bacterial caries models (`confidence=0.7387` via the calibrated negative Platt map) and rejected them as `does_not_qualify(wrong_system)`.
4. **CS4 — Unstated Host System Abstention (`artemisinic acid`, PMID `42418950`, noisy):** Describes *AaADH1* enzyme optimization for artemisinic acid production without specifying an engineered heterologous microbial host. *v2* forced a wrong `does_not_qualify`; *v3.2* verified `C1` and `C2` (`score=0.8562`) while holding `C3` unknown, abstaining cleanly as `insufficient_evidence(system_unstated, confidence=null)`.
5. **CS5 — Remaining Honest Failures (`42947235`, `42107802`, `PMC8123456`):**
   - `42947235` (real fungal lipase abstract): opens with a background sentence (*"Lipase-catalyzed biodiesel preparation..."*) triggered as a reagent pattern alongside true production sentences, routing conservatively to `insufficient_evidence(conflicting_evidence)` rather than auto-qualifying (`1` of `2` `qualifies` FN).
   - `42107802` (real Europe PMC abstract engineering flux toward dihydroartemisinic acid in an *"artemisinic acid-producing strain"*): `C3` matched umbrella *"engineered microbial platform"* rather than explicit `yeast`, emitting `does_not_qualify(wrong_system)` instead of abstaining cleanly (`gold=insufficient_evidence`).
   - `PMC8123456` (commercial Novozym 435 biocatalyst): yielded `insufficient_evidence(activity_unclear)` rather than definitive `does_not_qualify(reagent_role)` because NLI score fell inside the uncertain zone without an exact prepositional match.

---

## 5. Deliberate Deviations Register (D1–D6)

| ID | Topic | Normative Decision |
|---|---|---|
| **D1** | **Qualifier Platt Window** | Fit qualifier MLE on dev positives plus borderline negatives within `NEG_FIT_WINDOW = 0.15` below `tau` to eliminate upward slope bias (`A=3.4594`, `slope_unstable_flag=false`). |
| **D2** | **Verbatim Slice Invariant** | Hard assertion `canon_body[start:end] == evidence.display_text` at classification time; titles structurally isolated from evidence selection. |
| **D3** | **Externalized Constants** | All thresholds and weights pinned in [`config/constants.md`](file:///c:/Users/Ruhaan%20Kakar/Desktop/loopworm/literature-classifier/config/constants.md) (`splits_sha256` verified at load). |
| **D4** | **Strict Two-Veto Ceiling** | Only two rules may override supporting NLI spans: prepositional `reagent_role` (subject to **R3b** conflict check) and `review_article` (`pubType=review` when `scope == "this_study"`). |
| **D5** | **Conditional Null Confidence** | Only `qualifies`, `different_activity`, and `wrong_system` emit calibrated confidence; vetoes, `mention_only`, `background_citation`, and abstentions emit `confidence: null`. |
| **D6** | **Strict Abstract Contradictions** | Explicit contradictory host organisms in abstracts map to `does_not_qualify(wrong_system)` (deprecating `system_mismatch_abstract`), bounded by noisy abstention monitoring. |

---

## 6. Residual Limitations & Future Work (§8)

1. **Conservatism on Passive Negatives vs. Abstention Recall:** On the held-out test set, 8 `does_not_qualify` items routed to `insufficient_evidence` (`activity_unclear` or `system_unstated`), limiting 3-class `insufficient_evidence` precision (`0.0909`) and abstention error rate (`0.3333`) on `n = 3` true abstention cases.
2. **Thin Negative Platt Fitting Sample:** `negative_path` converged cleanly (`out_of_fold_brier = 0.1897`), but rests on only `n_rows = 4` `different_activity`/`wrong_system` dev rows; expanding annotated hard-negative dev rows is high priority before tightening negative band thresholds.
3. **Rule-Based Conflict Sensitivity:** Introductory sentences mentioning substrate/catalyst roles (e.g., `42947235`) still trigger **R3b** `conflicting_evidence` even when methods sentences describe native production, prioritizing curator safety (`1` total `qualifies` FP) over recall (`0.8947`).
4. **Abstract-Only Scope & Unparsed Criterion Tiers:** Without full-text PMC XML for paywalled publishers, host system details buried in Methods tables still trigger legitimate `system_unstated` abstentions. Additionally, **G1/G2/G3** cover unary roles, host constraints, exclusions, and numeric thresholds, but arbitrary multi-hop Boolean joins (`(A OR B) AND NOT C` across entities) intentionally return HTTP 422 rather than silently mis-scoring.
