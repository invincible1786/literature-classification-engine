<USER_REQUEST>
# implementation.md: v3.3 Surgical Fixes (post-live-probe, review-reconciled, FINAL)

**Goal:** fix the defects exposed by the live probe (`eval_runs/run_20261112_220742.json`) and
confirmed in code, without changing the architecture, TAU, or the 12-rule combiner. Then freeze and tag.

**Supersedes:** the earlier v3.3 draft and the v3.2 plan. Where any earlier document conflicts with
this file, this file wins. This version merges the two independent code reviews (general review and
critical-leak review); all of their accepted findings are folded into the steps below.

---

## Rules for this pass

1. **No threshold tuning.** Every fix is a gate, hypothesis, parser rule, or invariant.
2. **Fixes must be generic.** No paper IDs or entity names in code. Caffeine is a dev entity;
artemisinic acid and Pd(OAc)2 are test entities. After fixing against these queries they can no
longer serve as unbiased live evidence — they are regression fixtures only.
3. **Keep `--pipeline v2` runnable.** Keep `eval/splits.json` unchanged (its sha256 is recorded in
`config/calibration.json`). Because the file is byte-identical, **the splits_sha256 does not
change in this pass**; only `data_hash`, `fitted_at`, and the Platt (A, B) parameters in
`config/calibration.json` change after the F7 refit.
4. **Keep the §0 canonical spec table verbatim:** 12 rules, 18 registered codes, hard HTTP 422 on
unparsed clauses, LOEO on dev only. `tests/unit/test_doc_drift.py` parses the §0 table from this
file and asserts its numbers — do not reword, renumber, or move that table. No new rule or reason
code is needed; `entity_unclear`, `activity_unclear`, `possible_review`,
`unsupported_criterion_clause`, `different_activity` already exist. All F4–F13 behavior is
implemented **inside** the existing C1/C2/C3 outputs, the existing 12 rules, and the existing
reason registry. Rule-numbering note (for the docs): the combiner has 12 entries because R3 is
split into R3a/R3b; prose references "R1–R11 plus R3a/R3b" are the same 12.
5. **Any new constant gets a Constants Register entry** (in this file and mirrored in
`config/constants.md`), or CI fails.
6. **The title is never evidence.** Title text may be used for review-type classification (F13) and
exclusion-scope checks (F8), never to satisfy an evidence requirement.

---

## §0.5 Evaluation Contract (added per review; binding for all steps below)

1. **Record schema & provenance.** Frozen records use this stable schema:
`{pmid, doi, pmcid, id, title, abstractText, pubTypeList, source, retrieved_at, sha256}` where
`source` is `"europe_pmc"` and `sha256` covers a canonical serialization of all saved fields
(UTF-8, sorted keys, single JSON object, one record per line in the snapshot file). Duplicate
policy: dedup precedence `pmid → doi → pmcid → id`. Missing/changed-record policy: re-fetch once;
if still missing, exclude the record and log it in the manifest; never silently substitute.
2. **Criterion-specific annotation rubric.** Gold labels answer the *specific* user criterion
(route, system), not the generic "was the entity made" question. Each record carries per-condition
labels from the set `{supported, contradicted, unknown, not_required}` for entity, activity,
system, and route, each with an evidence span. The document-level `gold_3class` is derived by a
written mapping: `qualifies` iff all *required* conditions are `supported`;
`does_not_qualify` iff any required condition is `contradicted`;
`insufficient_evidence` otherwise (unknown / missing conditions). Disagreements are adjudicated
in a two-pass annotation with an adjudication log.
3. **Stratum isolation.** Live-audit labels live in **`eval/live_audit_labels.csv`**, never in
`eval/labels.csv`. `eval/run.py::load_dataset` default (`stratum_filter="all"`) selects only
`stratum in ("clean", "noisy")`; `live_audit` is returned only when explicitly requested. The
N=56 benchmark composition must not change.
4. **Metric definitions.** Macro-F1 is computed over **all three classes** regardless of presence in
the sample (fix the code/comment mismatch in `eval/metrics.py`). The v2 binary→3-class convention
(abstention → `does_not_qualify`) is a disclosed compatibility mapping only: v2/v3 three-class
comparisons must also report binary metrics separately. Abstention handling is stated per metric.
"Noise" is defined a priori: paired bootstrap confidence intervals on the same records; no
acceptance claim is made from raw point-estimate threshold crossings.
5. **Calibration spec.** Confidence means the held-out probability of the **predicted class**, not an
NLI margin and not a Platt fit on unvalidated ranking scores. Calibration is validated on held-out
predictions; report Brier score and reliability bins alongside ECE. LOEO fit stability is reported
but is not a substitute for held-out calibration quality.
6. **Canonical evaluation entry point.** `eval/run.py` is the only evaluator. `eval/evaluate.py`
(legacy, loads `eval/labeled.csv`) is quarantined/moved to `eval/legacy/`; the notebook invokes
`eval/run.py` with fixed seeds.
7. **Trace format.** Every prediction emits a trace linking: predicted conditions (entity, activity,
system, route), the selected evidence span with byte offsets, the parent source-sentence hash
(joining back to the frozen snapshot), parser output, and the gate/rule that produced the
decision. This powers invariant checks (g)–(k) and the final report.

---

## Root causes (confirmed in code and live trace)

| Symptom | Root cause | Fix |
| --- | --- | --- |
| "EPS synthesis", "caffeine and diterpene content" qualify | `split_clauses_bracket_aware` (conditions.py) splits sentences of 180+ chars at depth-0 commas; each fragment goes to DeBERTa and is returned standalone with no entity or verb check | F4, F5, F6 |
| 42107802 qualifies on dihydroartemisinic acid | Same: the returned clause lost the parent sentence's entity mention | F4, F5 |
| system=None for "engineered microbial biosynthesis" | criterion_parser.py only binds system after `in <system>`, so C3 stays not_required | F10 |
| "chemically synthesized" not enforced | No route/method requirement is parsed, so biological routes are never excluded | F11 |
| Degradation and bypass pass as synthesis | NLI hypotheses lack a degradation contrast; no structural guard for bypass | F7, F8 |
| "utilizing palladium acetate salt" qualifies | Missing pattern in role_patterns.json | F9 |
| Reviews leak when pubType is not review | Detection relies on the metadata tag and a narrow phrase check | F13 |
| Schema and slice invariants | Top-level evidence kept on abstentions; dedup_of="unknown"; whitespace collapsed after slicing | F1, F2, F3 |

**Latency is not a defect.** Q2's internal total_execution_ms was 2112 ms. The ~18 s gap was live
Europe PMC / PubChem network time. Do not investigate the model. Instead, log retrieval_ms
separately from total_execution_ms (F14) so this is visible next time.

---

## Step 0: Baseline (before touching code)

1. Commit current state on branch `v3.2-pre-fixes`.
2. `pytest tests/unit tests/model -v` must show 62 passed.
3. Copy `eval_runs/run_20261112_220742.json` to `eval/live_regression_v32.json` as the "before" file.

## Step 1: Freeze the 40 live records (the apples-to-apples baseline)

The run JSON holds only classifier output snippets, not raw abstractText or pubTypeList. The two
live runs also returned different PMIDs, so they are not comparable as they stand.

1. Fetch the full raw Europe PMC record per the §0.5 contract schema for each of the 40 paper_ids
**once**. Save to `data/snapshots/live_audit_40.json` with SHA-256 per record and a manifest
entry, matching the existing snapshot format.
2. Hand-label each record per the §0.5 rubric: `gold_3class` + per-condition entity, activity,
system, route labels with evidence spans. Save to **`eval/live_audit_labels.csv`** with
`stratum = live_audit` — **not** `eval/labels.csv` (leak-risk fix: entity-based filtering in
`eval/run.py` would otherwise silently swell the N=56 benchmark).
3. Update `eval/run.py::load_dataset` so default `"all"` means `stratum in ("clean", "noisy")`;
`live_audit` requires an explicit flag. Add a test that the N=56 benchmark row count is unchanged.
4. Hold out from calibration. `eval/calibrate.py` must exclude live-audit rows (by file provenance,
not only by entity name). Add a test: no live_audit row appears in any fit set (extends
`test_calibration_split_disjoint`).
5. Run `--pipeline v2` and the unmodified v3.2 over this exact frozen file. Record confusion
matrices, FP/FN counts, precision, recall, F1. This is the "before" for Step 8.

## Step 2: Output-contract fixes (low risk, do first)

| ID | Fix | File | Gate |
| --- | --- | --- | --- |
| F1 | When `evidence_status == "insufficient"`, set top-level `evidence`, `evidence_sentence`, `evidence_offsets` to null **in `sync_compatibility_fields`** (do not reconstruct). Keep the candidate span only under `conditions.*.evidence`. Read `tests/unit/test_v3_remediation.py` first and update any test that asserts non-null, noting the change in the commit message. | app/schemas.py | invariant (c) = 0 |
| F2 | `dedup_of` is null or a valid canonical paper_id, never "unknown". Resolution order: pmid → doi → pmcid → id. If the surviving record has no distinct valid ID, `dedup_of = None` (do not emit "unknown"). | app/retrieval.py | invariant (h) = 0 |
| F3 | Build raw_text as the exact slice of the canonicalized body. Collapse whitespace **before** slicing, so `canon_body[start:end] == display_text == clean(raw_text)` holds for multi-space inputs such as 10.21203/rs.3.rs-3926665/v1. | evidence slicer | invariant (a) whitespace = 0 |
| F14 | Add `retrieval_ms` to response timings, separate from `total_execution_ms`; verify the existing retrieval-path timing is combined and exposed in the same response object. | app/main.py | unit test on field |

## Step 3: Evidence selection (largest source of live false positives)

**F4. Clauses are auxiliary scoring units, not evidence — and the parent sentence is always the
public span.**

- Clause splitting (`CLAUSE_SPLIT_MIN = 180`, bracket-depth scanner) stays, but **only** for NLI
scoring.
- The returned evidence span is **always** the full parent Punkt sentence
`[sent.start:sent.end]`, byte-exact. No exceptions for punctuation, lowercase starts, or
mid-word cutoffs. This automatically recovers the parent entity mention on split sentences
(e.g., the dihydro-artemisinic collision) and eliminates all lowercase comma-fragment cutoffs.
- A clause is *scored* only if it (a) contains the entity or an alias, or has a resolved
antecedent in `anaphora_map`, and (b) contains a finite production verb. Clauses failing this
are skipped for scoring, never returned.
- The `MIN_EVIDENCE_CHARS = 30` + finite-verb quality check (F6) gates whether the parent
sentence may serve as evidence at all.

**F5. Entity-anchor gate (anaphora-safe).** A qualifies result requires that the returned evidence
span, or its resolved antecedent within `ANAPHORA_WINDOW = 2`, contains the entity or an alias.

- Word-boundary match.
- Reject a match where the entity is the tail of a longer compound. The prefix list is the
**single unified constant** previously split between `_RELATED_PREFIXES` (conditions.py) and the
proposed `COMPOUND_PREFIXES`: `[dihydro, deoxy, iso, nor, epi]`, configured in one place and
imported by both. "dihydroartemisinic acid" must not satisfy "artemisinic acid".
- Gate failure returns `insufficient_evidence(entity_unclear)`.
- Do not require the literal entity string in a sentence that has a resolved antecedent. "We
engineered yeast for X. It was produced at 25 g/L." must still qualify.
`test_ws14_e_cross_sentence_anaphora_gating` must stay green.

**F6. Minimum evidence quality.** Returned activity evidence must be at least `MIN_EVIDENCE_CHARS =
30` and contain a finite verb (verb-lexicon check). Otherwise abstain with
`insufficient_evidence(activity_unclear)`.

## Step 4: Activity polarity and role (semantic first, minimal regex)

**F7. Contrast hypotheses** in `h_alternatives` (conditions.py). Let DeBERTa reject degradation and
measurement semantically instead of piling up regexes. Add to the alternative set:

- "{entity} was degraded, metabolized, or broken down in this study."
- "{entity} content was measured or quantified in this study."

If an alternative hypothesis outscores the target hypothesis on the evidence span, the result is
`does_not_qualify(different_activity)`. This targets "degrading caffeine", "caffeine degradation
pathways", and measured-analyte papers.

**F8. Minimal structural guards only**, clause-scoped for *detection* but scope-checked against the
**parent sentence + title** (not only the isolated winning clause — an adjacent comma clause such
as "in tea plants" or "fermentation" must be caught):

- Bypass/negation: `without (going through|using|passing through) {E}`, `bypass*`, `not via`.
- Degradation stems: `degrad*`, `decaffein*`, `catabol*`.

On a hit, return `does_not_qualify(different_activity)`. Do not grow this list further; extend the
hypotheses instead. Failure of the lexicon on unusual phrasing is listed as a limitation.

**F9. Role patterns** (config/role_patterns.json, existing `{E}` placeholder):

- `utilizing|employing|using\s+(?:the\s+)?(?:[\w\-]+\s+){0,2}{E}`
- `{E}\s+(?:salt\s+)?as\s+(?:the\s+)?(?:precursor|catalyst|reagent|metal source)`

The reagent veto must not fire when the same abstract affirmatively reports synthesizing the entity
as the object. The existing R3b conflicting_evidence routing already covers this; add a fixture to
prove it.

## Step 5: Criterion parser (fixes system=None and unenforced routes)

**F10. Pre-nominal system adjectives bind spec.system.** Parse `<modifier>
(biosynthesis|production|synthesis|fermentation)`:

- microbial, engineered microbial, yeast, bacterial → `system.family = microbial`
- plant → `plant`
- cell-free, enzymatic, chemoenzymatic → `in_vitro`

C3 then runs instead of staying not_required.

**F11. Route requirement.** `chemically synthesized / chemical synthesis` sets `method = chemical`,
with automatic exclusions `{biosynthesis, fermentation, enzymatic, plant extraction, microbial,
in vivo}`. A document describing an excluded route returns `does_not_qualify(different_activity)`
through the existing exclusion-constraint branch of R8. Exclusion matching runs against the parent
sentence plus title (see F8 scope).

**Failure contract (unchanged).** A route or system word the parser recognizes but cannot enforce
returns HTTP 422 by default, and `insufficient_evidence(unsupported_criterion_clause)` under
`--allow-partial-criterion`. Never silently ignore it, and do not turn malformed requests into
silent abstentions.

**F12. Parser regression fixtures.** These must yield non-null constraints:

- "The compound was produced by engineered microbial biosynthesis in this study." →
`system.family = microbial`
- "Caffeine was chemically synthesized in this work." → `method = chemical` with the exclusion set
- IUPAC fixtures (2,2'-bipyridine, N,N-dimethyltryptamine, dibenzo[b,d]oxepine) still survive
clause splitting.

## Step 6: Review detection beyond pubType

**F13.** When pubType is not review, run a text-cue check on title and abstract (the title is used
for this classification only, never as evidence):

- **Strong cues:** "this review", "we review", "literature review", "here we summarize".
- **Weak cues (atomic generic phrases only — no multi-word title-specific strings):** "recent
advances", "perspectives", "clinical potential", "mechanistic insights", "biotechnological
advances", "current understanding", "future perspectives", "state of the art".
- **Results markers (veto blockers):** "we show", "we found", "we constructed", "was achieved",
"were synthesized".

Routing: strong cue with no results marker → `review_veto`; weak cues only → `possible_review`
(abstain). A results marker always blocks the veto, so research papers with "recent advances" in
the title are not lost. New constants (`REVIEW_STRONG_CUES`, `REVIEW_WEAK_CUES`,
`RESULTS_MARKERS`) go in the register and in config. Targets 41474478 and 38474639.

## Step 7: Strengthen the invariant checker

Extend `eval_runs/check_invariants.py` and wire it into CI:

| ID | Check |
| --- | --- |
| (g) | Every qualifies evidence span contains the entity or an alias at a word boundary, or the response exposes a resolved antecedent that does. No compound-prefix collision. The API must expose the antecedent span in diagnostics so the checker can verify anaphoric qualifiers. |
| (i) | Every returned evidence span is a complete parent sentence boundary — not a lowercase comma fragment or a clipped word. |
| (j) | No qualifies evidence contains a degradation or bypass cue (F8). |
| (k) NEW | Every evidence span joins back to the frozen snapshot: record sha256 matches and `canon_body[start:end]` reproduces the stored span exactly (source-record join + evidence-offset verification). Includes the antecedent diagnostic. |

Invariants (a)–(k) must all be 0.

## Step 8: Verification and reporting

1. **Unit and model tests.** `pytest tests/unit tests/model -v` passes. ~15 new tests, one per
fix, plus: dihydro-prefix collision, anaphora still qualifies, fragment not returned
standalone, R3b with `utilizing {E}` plus an affirmative synthesis claim, review cues blocked
by a results marker, F12 fixtures, no live_audit rows in calibration fit, N=56 row count
unchanged.
2. **Doc-drift CI.** Still green: `len(RULES) == 12`, `len(REASON_REGISTRY) == 18`, combiner
literals all in the registry, §0 table in this file untouched verbatim.
3. **Calibration.** F7 changes the NLI margins, so refit on dev only (`eval.calibrate`, LOEO).
Because `eval/splits.json` is unchanged, **the recorded splits_sha256 stays identical**; update
`data_hash`, `fitted_at`, and Platt (A, B) only. Validate actual fit-row provenance and saved
calibration metadata; reject the refit if any fit row is a test entity or live_audit row.
Report ECE, Brier, reliability bins, and LOO spread.
4. **N=56 benchmark.** Re-run `notebooks/evaluation.ipynb` (via `eval/run.py`, fixed seeds) with
the real DeBERTa scorer. Acceptance: qualifies precision ≥ 0.94, recall ≥ 0.89 (no drop beyond
the pre-registered noise bound), 3-class macro-F1 over all classes ≥ 0.6381, ECE ≤ 0.0695 +
noise. All acceptance judgments use paired bootstrap CIs on the same records. If recall drops,
find which gate over-fired and fix the gate. Do not loosen it blindly.
5. **Frozen live audit (40).** Run v2, v3.2, and v3.3 on `live_audit_40.json`. Report confusion
matrices, FP/FN counts, precision, recall, F1 (binary and 3-class separately, with the v2
mapping disclosed), paired CIs, and the per-paper flips with the gate that caused each.
Acceptance is judged against gold labels, not against expected counts. Directional expectation:
Q2 and Q3 false positives drop sharply and Q4's single FP disappears.
6. **Fresh live probe (the only unbiased live number).** Pre-register 2–3 new entities (for
example resveratrol, naringenin) in `eval/live_probe_entities.json`, chosen **before** any live
run; they must not appear in `splits.json` (its hash stays valid). Add a disjointness test.
Run top-10 live queries with a fixed retrieval procedure, freeze the records exactly as in
Step 1, hand-label **all** retrieved results (or a pre-specified sample), and report precision,
recall, FP examples, **plus retrieval coverage and the denominator** — not just top-10 numbers.
7. **Live re-run on the same four queries** with `check_invariants.py`: all of (a)–(k) are 0.
Report latency as retrieval_ms versus total_execution_ms; no investigation of the model is
needed.

## Step 9: Freeze

1. Logical commits: contract/eval-infra fixes, output contract (F1–F3, F14), evidence selection,
polarity/role, parser, review cues, tests/docs.
2. Update `KNOWN_ERRORS.md` with any failures remaining after Step 8. Update
`TECHNICAL_REPORT.md` with the frozen-live-audit before/after table, the fresh probe result
(with coverage and denominator), the calibration report (ECE + Brier + reliability bins), and
the latency note.
3. Move `eval_runs/` to `eval/live_runs/` (or `.gitignore` it and keep the summary in the report).
4. Tag `v3.3-frozen`. Do not reuse `v3.2-frozen`, because the code changed.

---

## Constants Register additions

| Constant | Value | Notes |
| --- | --- | --- |
| MIN_EVIDENCE_CHARS | 30 | F6 |
| COMPOUND_PREFIXES | dihydro, deoxy, iso, nor, epi | F5; **unified with `_RELATED_PREFIXES`** — one configured constant, imported by both conditions.py and the entity-anchor gate |
| REVIEW_STRONG_CUES / REVIEW_WEAK_CUES / RESULTS_MARKERS | lists per F13 | atomic generic phrases only; config file + register |
| ANAPHORA_WINDOW | 2 | unchanged |
| CLAUSE_SPLIT_MIN | 180 | unchanged; NLI scoring only, never returned as evidence |

## Definition of Done

- [ ] Invariants (a)–(k) are all 0 on the live re-run.
- [ ] Offline tests pass (62 existing plus ~15 new); anaphora test still green; doc-drift test green.
- [ ] N=56 benchmark within the acceptance bounds, judged with paired CIs; row count unchanged.
- [ ] Frozen live audit (40) labeled in `eval/live_audit_labels.csv`, held out of calibration, with
v2 / v3.2 / v3.3 comparison on identical abstracts.
- [ ] Fresh live probe pre-registered, labeled, and reported with its own precision, recall,
coverage, and denominator.
- [ ] Calibration refit on dev only; splits_sha256 unchanged; data_hash, fitted_at, (A,B) updated;
ECE, Brier, reliability bins, LOO spread reported.
- [ ] Macro-F1 computed over all three classes; v2 binary mapping disclosed; canonical evaluator
(`eval/run.py`) is the only path invoked by the notebook.
- [ ] Docs updated (`KNOWN_ERRORS.md`, `TECHNICAL_REPORT.md`); `v3.3-frozen` tagged.

## Out of scope (list as limitations)

- Full-text retrieval for abstract-only abstentions.
- A learned review classifier.
- Calibration sets larger than the current dev pool (40 live rows are too few for Platt, and are
held out anyway).
- Dependency-parse negation scope. F8 is a heuristic and will miss unusual phrasing; F7 covers the
semantic side.
- Title-only cues, antecedent evidence, and evidence split across sentences: the current
evidence policy is abstract-only, parent-sentence spans; these cases are documented as
abstention causes rather than supported positives.

Implement
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-13T18:13:49+05:30.

The user's current state is as follows:
Active Document: c:\Users\Ruhaan Kakar\Desktop\loopworm\todo.txt (LANGUAGE_UNSPECIFIED)
Cursor is on line: 1
</ADDITIONAL_METADATA>