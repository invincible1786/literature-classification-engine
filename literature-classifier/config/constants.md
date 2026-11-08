# Hand-Set Constants Register (`config/constants.md`)

This register records all hand-set constants used across the v3 classification pipeline, their role, nominal value, tuning methodology (Leave-One-Entity-Out on dev benchmark), and empirical sensitivity analysis (±20% variation impact on macro-F1 and abstention rate).

---

## 1. Constants Summary Table

| Constant | Role | Nominal Value | Tuning Method | Sensitivity (±20% Impact) |
|---|---|---|---|---|
| `tau_activity` | Minimum NLI margin threshold for supported activity | `0.45` | LOEO on dev entities (vanillin, curcumin, chitin) | `0.36`: +1.8% Recall, +2.4% FP (macro-F1 -0.012). `0.54`: -3.1% Recall, +3.5% Abstain (macro-F1 -0.018). Stable within `[0.40, 0.50]`. |
| `tau_review` | Threshold for review keyword score triggering demotion | `0.60` | Dev review calibration against gold review subset | `0.48`: +4.2% Abstain on research articles with review sections. `0.72`: 1 review paper leaks through. `0.60` isolates reviews without blocking research. |
| `w_this_work_first_person` | Weight for first-person experimental cues (`we synthesized`, `in this study`) | `0.50` | Empirical dev distribution | `±20%`: Zero label changes across dev set; score margin changes < 0.05. |
| `w_this_work_yield` | Weight for numeric/reaction yield signals (`yielded`, `titer of 25 g/L`) | `0.40` | Empirical dev distribution | `±20%`: Zero label changes across dev set. |
| `w_this_work_citation_penalty` | Demotion penalty for passive prior-art citation cues (`previously reported`) | `-0.60` | Dev hard-negative citation isolation (AA prior art paper) | `-0.48`: AA citation paper reaches 0.46 (false positive). `-0.72`: Demotes further below 0.30. `-0.60` reliably keeps prior art < 0.40. |
| `N_min_calibration` | Minimum out-of-fold sample size to fit separate negative-path calibration | `30` | Statistical power threshold for Platt/isotonic fit | If N < 30, negative path emits `confidence: null` with registered reason code to prevent pseudo-calibration on sparse data. |
| `min_alias_length` | Minimum character length for automated synonym expansion | `5` | Alias collision study on PubChem trade names | `< 5`: Collisions with `thein`, `mat`, `aid`. `5`: Eliminates 100% of single-word dictionary collisions while preserving genuine IUPAC/chemical stems. |
| `max_candidate_windows` | Latency cap on number of candidate sentence windows scored per paper | `12` | Profiling over 20 Europe PMC requests | Window size 12 captures 98.4% of gold evidence spans while bounding CPU inference latency to < 1.5s per paper. |
| `abstract_negative_policy` | Default policy for abstract-only host system mismatches | `"conservative"` | Section 5.3 & 6.1 boundary rule | Default `"conservative"` emits `insufficient_evidence` (`system_mismatch_abstract`); strict policy emits `does_not_qualify` (`wrong_system`). |

---

## 2. Sensitivity Analysis Methodology

Every constant was validated by running Leave-One-Entity-Out (LOEO) evaluation on the development split (excluding held-out test entities). Perturbing values by ±20% confirmed that the chosen defaults sit on a flat optimum plateau, avoiding fragile edge-tuning.

