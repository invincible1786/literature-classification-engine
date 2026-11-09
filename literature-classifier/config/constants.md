# Hand-Set Constants Register (`config/constants.md`)

This register records all hand-set constants used across the v3.2 classification pipeline, their role, nominal value, tuning methodology (Leave-One-Entity-Out on `eval/splits.json` dev entities), and empirical sensitivity analysis (±20% variation impact on macro-F1 and abstention rate).

---

## 1. Constants Summary Table

| Constant | Role | Nominal Value | Tuning Method | Sensitivity (±20% Impact) |
|---|---|---|---|---|
| `TAU` (`tau_activity`) | Minimum NLI margin threshold for supported activity | `0.45` | LOEO on dev entities (`eval/splits.json`) | Test sensitivity: `0.36` (`F1=0.6474, FP=2, abst=0.1607`); `0.45` (`F1=0.6381, FP=1, abst=0.1964`); `0.54` (`F1=0.6249, FP=1, abst=0.2143`). |
| `DELTA` | Role-score dominance margin | `0.00` | Dev-tuned contrast margin | Stable ±20% across dev folds. |
| `PLATT_A`, `PLATT_B` | Qualifier-path Platt logistic MLE parameters | `MLE-fitted` (`config/calibration.json`) | LOEO on dev qualifiers + `NEG_FIT_WINDOW` | Refit per LOEO fold; LOO spread reported. |
| `PLATT_A_NEG`, `PLATT_B_NEG` | Negative-path Platt logistic MLE parameters (`different_activity`, `wrong_system` only) | `MLE-fitted` (`config/calibration.json`) | LOEO on dev evidence-based negatives | Refit per LOEO fold; LOO spread reported. |
| `NEG_FIT_WINDOW` | τ-centered window for including borderline negatives in qualifier Platt fit | `0.15` | Slope conditioning on LOEO dev split | LOEO check prevents upward slope bias. |
| `ANAPHORA_WINDOW` | Antecedent lookback sentence window for pattern-based coreference | `2` | Structural syntactic span test | Unit-tested across 1–2 sentence lookback. |
| `CLAUSE_SPLIT_MIN` | Character length threshold triggering bracket-depth clause splitting | `180` | Replaces removed `MAX_SENT_CHARS` (WS15-B) | ±20% LOEO sensitivity checked in notebook. |
| `CLAUSE_SPLIT_PATTERNS` | Depth-0 priority split boundaries inside bracket scanner | `;`, `, however,`, `, whereas,`, `, although,`, `, but `, `, ` | Fixed grammatical priority order | Bracket scanner preserves IUPAC names cleanly. |
| `P_THIS_WORK_MIN` | Clause-local experimental attribution gate threshold | `0.50` | Dev hard-negative citation isolation | Stable ±20%; prior-art clauses without positive cues score < 0.40. |
| `tau_review` | Review lexical score threshold triggering demotion | `0.60` | Dev review calibration against gold review subset | Isolates reviews while preserving primary research papers. |
| `w_this_work_first_person` | Clause-local weight for first-person experimental cues (`here we`, `in this study`) | `0.50` | Empirical dev distribution | ±20%: Zero label changes on clean dev slice. |
| `w_this_work_yield` | Clause-local weight for numeric/reaction yield signals | `0.40` | Empirical dev distribution | ±20%: Zero label changes on dev set. |
| `w_this_work_citation_penalty` | Demotion penalty for prior-art citation cues (`previously reported`) | `-0.60` | Dev hard-negative citation isolation | Keeps prior-art-only clauses < 0.40. |
| `band_edges` | Qualitative confidence band cutoffs (High / Medium / Low) | `0.80 / 0.50` | Monotone probability thresholds | Monotone by construction. |
| `ambiguity_margin` | Minimum top-2 margin gap before High → Medium confidence dampening | `0.10` | Dev calibration ambiguity guard | Stable ±20%. |
| `min_alias_length` | Minimum character length for automated synonym expansion | `5` | PubChem collision audit | Eliminates short-word collisions (`thein`, `aid`, `mat`). |
| `max_candidate_windows` | Latency cap on number of candidate sentence windows scored per paper | `12` | Profiling over Europe PMC requests | Bounds latency while retaining recall. |
| `splits_source` | Single source of truth for dev/test entity separation | `eval/splits.json` | Audited via `splits_sha256` in `config/calibration.json` | Hard failure on train/test overlap or hash drift. |
| `MIN_EVIDENCE_CHARS` | Minimum character length for valid activity evidence span | `30` | F6 evidence quality gate | Rejects short comma fragments (e.g. 'EPS synthesis'). |
| `COMPOUND_PREFIXES` | Chemical prefix list for compound collision rejection | `dihydro, deoxy, iso, nor, epi` | F5 unified prefix register | Rejects matches where entity is derivative tail. |
| `REVIEW_STRONG_CUES` | Unambiguous review text cues in title or abstract | `this review, we review, literature review, here we summarize` | F13 review detection | Triggers review_veto when results markers absent. |
| `REVIEW_WEAK_CUES` | Suggestive review phrase cues in title or abstract | `recent advances, perspectives, clinical potential, mechanistic insights, biotechnological advances, current understanding, future perspectives, state of the art` | F13 review detection | Triggers possible_review when results markers absent. |
| `RESULTS_MARKERS` | Empirical results phrases blocking review veto | `we show, we found, we constructed, was achieved, were synthesized` | F13 veto blocker | Prevents primary research with 'recent advances' from demotion. |

---

## 2. Removed Constants (v3.0 → v3.2)

- ~~`MAX_SENT_CHARS` (`250`)~~: Removed in **WS15-B**; replaced by bracket-depth syntactic clause splitting (`CLAUSE_SPLIT_MIN = 180`).
- ~~`abstract_negative_policy = "conservative"`~~: Superseded by **D6 (WS14-C)**; explicit abstract host contradiction defaults to `does_not_qualify(wrong_system)`.
