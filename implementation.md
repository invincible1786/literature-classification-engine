implementation.md — v2 → v3 Upgrade Plan (Revision 3)

Project: Mini Literature Classification Engine Goal: Move from a rule-gated, single-sentence, binary classifier (v2) to a criterion-aware, multi-condition, three-class classifier with validated evidence and honestly validated confidence (v3), extending the existing codebase rather than rebuilding it.

Derived from the project document, not a code audit. Verify each problem exists in the repo before fixing it; file and function names may differ. Status vocabulary: an issue is Planned until its test or metric (see the tracker at the end) passes, then Verified. Nothing is called "Addressed" before that.

Revision 3 Log (changes from the previous plan)

Earlier revisions (reviewer-case gate, contamination control, frozen-code protocol, cluster bootstrap, 3×3 scoring, parser clause tracking, model probe, reason-code registry, API versioning) are retained. This revision fixes problems found in review.

#	Problem found	Fix	Where
R1	Rules 4, 8, 9 were first-match hard negatives from NLI/string output, contradicting "exactly two vetoes" and hurting recall	Separate vetoes (may override support) from evidence-based negatives (never override support, need a verbatim span, carry abstract-only carve-outs)	0 (P1), 6.1
R2	entity_absent clashed with the boundary rule (entity may be named only in Methods)	entity_absent → does_not_qualify only with full text; abstract-only → entity_unclear	5.1, 6.1
R3	wrong_system could reject a paper whose abstract names only E. coli while full text used yeast	Abstract-only wrong_system emits insufficient_evidence (system_mismatch_abstract) by default; a strict policy is reported as a sensitivity row	5.3, 6.1
R4	Any conflict abstained the whole paper	Conflicts demote one condition to unknown; a paper naming two organisms/activities is not a conflict if the requested one is supported (also_mentions)	5.6, 6.1
R5	review_score was computed but unused	Wired in: high review_score with low p_this_work demotes activity to unknown (possible_review)	4.4, 5.4
R6	Evidence-containment deviation was silent	Documented as Deliberate Deviation D1; spans carry role, sentence index and verifiable_alone	4.2, Deviations
R7	Unsupported clauses both "rejected" and "abstained"	One policy: 422 by default; opt-in allow_partial_criterion evaluates the core, can reject, cannot qualify	3.3
R8	Rule 7 sat above rules 8–9	Unsupported-clause cap moved below the negatives	6.1
R9	Graded nDCG gain (insufficient = 1) was circular	Binary nDCG primary; tier order for non-qualifiers decided empirically on dev; graded gain only a flagged sensitivity	7.5
R10	Per-path, per-stratum calibration too thin for 150–200 papers	Single qualifier-path fit of P(correct); negatives fitted only if ≥ N_min, else confidence: null; evidence_source reported descriptively	7.2
R11	CriterionSpec didn't fit G2	entity_role ∈ {product, agent}, agent_kind, optional activity; agent-role activity check defined	3.1, 5.2
R12	Response schema missing promised fields	Added aliases_used, calibration_version, retrieved_by, evidence raw_text/display_text/offsets/role	6.8
R13	Timeline couldn't hold the ≥1 week relabel gap	Calendar with explicit day numbers; Phases 2–5 fill the gap	1.3, Order
R14	Labeling budget unrealistic	Two-tier labeling, minimum viable benchmark, realistic hours	1.1, 1.4
R15	Agentless-passive fail-safe not stated	Stated in 5.4	5.4
R16	Ranking comparison didn't isolate reordering	Three ranking variants compared	7.5
R17	Garbled Principle 3	Rewritten	0 (P3)
0. Guiding Principles
Two kinds of negative, kept distinct.
Vetoes (exactly two) may override supporting evidence because they are rule-provable: (a) explicit prepositional role patterns where the entity is the object of the role preposition, (b) Europe PMC pubType=review with no first-person experimental cue.
Evidence-based negatives (different_activity, wrong_system, entity_absent) come from NLI or string matching. They never override supporting evidence, require a verbatim contradicting span, and have abstract-only carve-outs (6.1).
Three outcomes. qualifies, does_not_qualify, insufficient_evidence. "I don't know" is never merged with "no".
unknown is not zero. A condition blocks qualifies only if it is contradicted, or if it is required and unknown. unknown never converts into does_not_qualify.
Evidence is a pointer into the raw source. Never generated, trimmed, paraphrased or joined.
Claims match the data. Small benchmark → approximate confidence, wide CIs, stated limitations.
No tuning on test or on the reconstructed reviewer proxy.
Fix the approach, not the thresholds. Every hand-set constant is registered and disclosed (5.0).
Scope honestly. Unfixed issues get a measured impact, not a vague "future work".
Deliberate Deviations (stated, not hidden)
ID	Deviation	Rationale	Mitigation
D1	Only the activity span must contain the entity or alias; entity and system spans may not	A system sentence ("We engineered a novel S. cerevisiae strain…") rarely names the compound	Each span carries role, sentence_index, verifiable_alone; system span must lie within the selected window (≤2 sentences from the activity span, same section); limitation stated
D2	Evidence-based negatives exist beyond the two vetoes	Dr. Reddy's "related but different activity" case needs a negative class	Never override support; need a verbatim span; abstract-only carve-outs; sensitivity row
D3	Unsupported clauses can run only under opt-in allow_partial_criterion, and then cannot qualify	Avoids guessing while still allowing rejection of clearly wrong papers	Default is 422
D4	Negative-path confidence is usually null	Too few predicted negatives to fit honestly	Reason code carries the explanation
Phase 0 — Baseline, Reviewer Case, Reproducibility
 0.0 Reviewer-case gate.
Ask Dr. Reddy for the exact entity, criterion and flagged papers. Check his criterion against the Phase 3 grammar before the grammar is frozen.
Fallback: if no reply in 3–4 days, build reviewer_case_reconstructed from his email, labeled as a reconstruction in every table and excluded from tuning. If the real case arrives later, run it once as a final check.
Run v2 on it and save the failing output before any v3 code is written.
 0.1 Tag v2 (git tag v2-baseline); keep it runnable with --pipeline v2.
 0.2 Pin everything: model revision, exact requirements.txt, seeds.
 0.3 Snapshot external data (Europe PMC records and searches, PubChem synonyms) with a SHA-256 per raw record. Add --offline that forbids network calls.
 0.4 Record v2 predictions on the benchmark and reviewer case.
 0.5 Profile latency per stage over ~20 requests. Nothing in Phase 10 is optimized without this.

Done when: --pipeline v2 --offline reproduces identical predictions twice.

Phase 1 — Evaluation Foundation
 1.1 Two-tier labeling in eval/labels.csv.
Tier 1 (every paper): label3, reason_code, scope_this_study, one-line rationale.
Tier 2 (subset: all gold qualifies, all hard negatives, all insufficient_evidence boundary cases, ≥40 papers): cond_entity, cond_activity, cond_system (yes/no/unclear) and gold evidence (gold_evidence_text, gold_raw_start, gold_raw_end, gold_evidence_source).
Candidate recall (4.6) is measured on Tier 2 only; the report states this.
 1.2 Operational boundary rule (in the file header and the report):

insufficient_evidence: the text provided does not decide the question, and a Methods/Results-style sentence in the full text would plausibly contain the answer. does_not_qualify: the text affirmatively shows a different activity, role, system, or a non-experimental paper type.

 1.3 Self-relabel with a real calendar.
Day 0: finish Tier 1 labeling in one sitting, SHA-256 labels.csv, commit. No v3 design before this.
Days 1–8: Phases 2–5 (offsets, parser and probe, candidates, conditions). These do not depend on label values.
Day ≥ 8: blind re-label of ≥30 papers (≥7 days after first pass; if Phases 2–5 finish early, continue with Phase 6 drafting, not with labels).
Report intra-rater Cohen's κ per class and on reason_code, labeled intra-rater, never inter-annotator.
 1.4 Benchmark size, realistic.
Minimum viable: the existing 100 papers (Tier 1) + ≥40 Tier 2 + the reviewer case. Estimated effort: Tier 1 ≈ 5 min/paper (~8 h for 100), Tier 2 ≈ +10 min/paper (~7 h for 40): about 3 days.
Stretch: grow toward 150–200 papers across ≥5 entities, including one organism-specific criterion and one agent-role criterion, if time allows.
Report the achieved n, never the target.
 1.5 Split and freeze protocol.
Split by entity into dev and test; commit eval/splits.json.
Development, tuning and calibration use leave-one-entity-out (LOEO) on dev only. test entities are never inspected.
Code freeze: git tag v3-frozen plus config hash. No changes afterwards.
Headline: one run of pooled dev+test LOEO (calibration refit per fold, no code changes); separately test-only with dev-fit parameters as robustness. The report states which is primary and that it ran once.
CIs use a cluster bootstrap over entities; Wilson is shown only as a naive reference.
 1.6 Leakage check: the 30-paper audit and N1–N6 fixtures are not in test.
 1.7 Metrics (eval/metrics.py): per-class and macro P/R/F1, 3×3 and binary confusion matrices, FP/FN, abstention and overclaim rates, coverage-vs-accuracy curve, cluster-bootstrap CIs, exact McNemar (only with enough discordant pairs), Brier, ECE with bootstrap CI.
 1.8 Scoring matrix (strict and lenient always both). Positive class = qualifies.
Gold \ Predicted	qualifies	does_not_qualify	insufficient_evidence
qualifies	TP	FN	Strict: FN. Lenient: excluded, coverage loss
does_not_qualify	FP	TN	Strict: abstention rate only. Lenient: excluded
insufficient_evidence	FP, also overclaim rate	TN	agree
Also report macro-F1 over all three classes.

Done when: v2 yields the full metric table (including recall and F1) under both conventions.

Phase 2 — Evidence Integrity
 2.1 Offset-preserving canonicalizer (text_utils.py). Segments (canon_start, canon_end, raw_start, raw_end) recorded per operation; map_span snaps outward to enclosing raw boundaries. Offsets refer to the hashed raw record; responses include source_record_sha256.
 2.2 Return both forms: raw_text = raw[raw_start:raw_end] (verbatim, may contain markup) and display_text (markup stripped). Never trimmed, paraphrased or joined. Offsets are character (code point) offsets, not bytes.
 2.3 Round-trip invariant: normalize(raw[raw_start:raw_end]) equals the canonical slice.
 2.4 Section-aware segments (best effort). Segment{text, source, raw_start, raw_end}, source ∈ {abstract, intro, methods, results, discussion, caption, title}. Parse JATS only if time remains after Phases 0–7; otherwise abstract-only and evidence_source says so.
 2.5 Titles are never evidence (schema validator and test). Titles serve retrieval, entity detection and the metadata review check only.
 2.6 Sentence-split tests: chemical names, 25.4 g/L, sp. nov., e.g..

Tests: property test on random benchmark abstracts and spans; ligatures, super/subscripts, combining characters, nested <i>/<sub>, &amp;; test_title_never_returned_as_evidence.

Phase 3 — Criterion Parser (Scoped) and Model Probe
 3.1 Schema (fits both grammar shapes):
python
  class CriterionSpec(BaseModel):
      entity: str
      entity_role: Literal["product", "agent"]
      agent_kind: Optional[Literal["catalyst", "reagent", "substrate"]]  # required iff role == "agent"
      activity: Optional[str]          # required iff role == "product"; optional for agent
      system: Optional[SystemSpec]     # None => not required
      scope: Literal["this_study", "any"] = "this_study"
      unparsed_clauses: list[str] = []
      parse_method: Literal["rules", "user_supplied"]
      parse_confidence: float

A validator enforces the two "required iff" rules.

 3.2 Enumerated grammar (README and report):
G1 (product): <paper> reports <activity> of <entity> [in|using|by <system>] [in this study]
G2 (agent): <paper> uses <entity> as a <catalyst|reagent|substrate> [in <activity>] → entity_role = agent, agent_kind set, activity optional.
A few documented paraphrases per shape. Anything else → 422 listing supported forms.
 3.3 Unsupported-clause policy (one rule).
Default: a recognized-but-unsupported clause (e.g. "with dose-dependence") → 422; it is also listed in unparsed_clauses in the error body. Nothing is guessed.
Opt-in allow_partial_criterion=true: the supported core is evaluated and the clause is recorded in unparsed_clauses. The paper can be rejected (negatives below), but cannot be qualifies; where it would have qualified, the result is insufficient_evidence / unsupported_criterion_clause. (Deviation D3.)
 3.4 Role polarity. The role veto and negative-role logic apply only when entity_role == "product". For agent, role patterns become supporting evidence.
 3.5 Systems gazetteer config/systems.json: organisms with synonyms and families (yeast ⊃ S. cerevisiae, P. pastoris; microbial ⊃ yeast, bacteria, fungi).
 3.6 Override and echo. POST /classify accepts criterion_spec; every response echoes the parse.
 3.7 Model behavior probe (1 day, before 5.0 cut-offs and the 6.1 table are finalized). ~30 hand-written premise–hypothesis pairs on negation, hedging, neutral-vs-contradiction, multi-sentence premises, numeric yields, catalyst-vs-product phrasing. Save eval/probe_results.md.

Tests: ≥15 supported criteria (organism-specific, organism-agnostic, agent-role), ≥5 unsupported shapes (→ 422), an extra clause appears in unparsed_clauses, partial-criterion mode can reject but never qualify, validator rejects a G2 spec without agent_kind.

Done when: "…biosynthesis of X in yeast" and "…of X" yield different specs and different outcomes on an E. coli paper; the reviewer criterion (or proxy) parses or is explicitly rejected.

Phase 4 — Soft Candidate Generation
 4.1 Open vocabulary (Issue 12). A sentence is a candidate if the entity (or resolved coreference) appears and/or an activity signal fires: verb families, nominalizations (production, titer, yield, accumulation, heterologous expression), passives, plus a weaker NLI/embedding relevance signal.
 4.2 Windows select candidates only (Issue 2).
Windows of 1–3 consecutive sentences (same section) are scored to choose candidates.
Returned evidence is verbatim single-sentence spans with role ∈ {entity, activity, system}, sentence_index, and verifiable_alone. Windows never appear as evidence; text is never joined.
Containment (Deviation D1): the activity span must contain the entity or a vetted alias. Entity and system spans need not; the system span must lie within the selected window (≤2 sentences from the activity span, same section) and is flagged verifiable_alone: false if it lacks the entity.
If the activity sentence refers to the compound only as "the product" and no sentence names it with the activity → insufficient_evidence / evidence_split.
Each accepted span passes the single-sentence check in 5.5.
 4.3 Light coreference (rule-based): "this compound", "the product", "the metabolite", "the target molecule" resolve to the entity only when it appeared in the immediately preceding sentence. Resolved coreference supports candidate selection, not the containment rule.
 4.4 Review handling (Issue 14), two tiers.
Veto: pubType indicates review and no first-person experimental cue and scope == this_study → does_not_qualify / review_article.
Soft: keyword hits ("we review…") produce review_score, which is consumed in 5.4 (not orphaned).
 4.5 Candidate cap (~12 windows per paper after cheap prefilter).
 4.6 Gate recall at span level on Tier 2 papers: does any candidate sentence overlap the gold evidence span? Target ≥95%; list misses.

Tests: the two-sentence yeast example yields a valid span set; nominalization-only sentence is a candidate; a research abstract with "we review" is not auto-abstained; evidence_split fixture abstains; system span without entity is flagged verifiable_alone: false.

Phase 5 — Condition Checkers
5.0 Hand-set constants register

Every constant lives in config/constants.md with its role, value, how it was chosen (LOEO on dev), and a sensitivity analysis (±20%, effect on F1 and abstention rate). Includes: activity margin cut-offs, τ_review, attribution weights (hand-set, disclosed), N_min for calibration cells, and the abstract_negative_policy flag (5.3).

python
class ConditionResult(BaseModel):
    name: Literal["entity", "activity", "system"]
    status: Literal["supported", "contradicted", "unknown", "not_required"]
    score: float                    # raw model/feature score, NOT a probability
    evidence: list[EvidenceSpan]
    also_mentions: list[str] = []   # e.g. other organisms/activities named, not in conflict
    reason_code: Optional[str]
 5.1 C1 Entity.
Alias matching with lookarounds; default minimum length 5; short aliases allowed when validated by co-occurrence with the full name in-document; author/affiliation fields excluded; related-compound guard ("dihydroartemisinic acid" ≠ "artemisinic acid").
Status rules (fixes the abstract-only clash): supported if matched in an eligible span; contradicted (entity absent) only when full text is available and the entity is absent from it; with abstract-only text and no match → unknown (entity_unclear), because Methods may name it.
Role veto (hard): only when entity_role == "product" and the entity string is the object of an explicit role preposition ("in the presence of {entity}", "catalyzed by {entity}", "using {entity} as catalyst"). "{entity} was produced in the presence of Y" must not trigger it. → reagent_role.
InChIKey/ChEBI normalization is best-effort, disk-cached, never required at request time.
 5.2 C2 Activity.
Product role: competing-hypothesis NLI over the requested activity vs. alternatives (chemical synthesis, extraction, degradation, pharmacology, clinical, computational). 3-way output; neutral = unknown. supported if the requested activity wins by margin; contradicted if a different activity wins with a verbatim this-work span and the requested activity has no support; else unknown. Mean over faithful paraphrases.
Agent role (G2): hypothesis "{entity} is used as a {agent_kind} [in {activity}] in this study"; alternatives are "{entity} is the product", "{entity} is the analyte/target", "{entity} is only mentioned". Role patterns count as supporting evidence here. contradicted only if an alternative wins with a verbatim this-work span.
Reason codes: different_activity, mention_only, hedged, negated, background_citation, possible_review, conflicting_evidence.
 5.3 C3 System.
Gazetteer plus NLI "produced in {system}". supported if the requested system (or family member) has this-work evidence. Other organisms named are listed in also_mentions, not treated as contradictions.
contradicted only when a different system has this-work production evidence and the requested system has none.
unknown when no system is stated (common in abstracts); not_required when the spec has no system.
Abstract-only carve-out: the abstract may name only E. coli while Methods used yeast. Under abstract_negative_policy = conservative (default), an abstract-only contradicted is emitted as insufficient_evidence / system_mismatch_abstract. Under strict, it becomes does_not_qualify / wrong_system. Both are reported as a sensitivity row; full-text contradicted always gives wrong_system.
 5.4 Attribution and review as soft features, wired to scope.
Rule-weighted p_this_work from first-person cues, citation markers, "previously reported", "herein", tense, section when known.
Fail-safe: agentless passives ("X was produced in yeast") and mixed sentences resolve to unknown, never to a rejection.
Low p_this_work demotes supported → unknown (background_citation).
Review feature consumed here: review_score ≥ τ_review and low p_this_work demotes activity supported → unknown (possible_review). Never a veto.
All of 5.4 is off when scope == any.
 5.5 Evidence verification. Re-run NLI on each selected single sentence against its condition; demote to unknown if it no longer entails.
 5.6 Conflict handling is condition-level.
A conflict is supporting and contradicting evidence about the same claim (e.g. "produced in yeast" vs. "no production in yeast"; direct negation; dueling spans within margin).
Effect: that condition only is demoted to unknown with conflicting_evidence. The paper is not abstained as a whole by this rule; the final label follows 6.1 from the demoted condition.
Two organisms or two activities where the requested one is supported is not a conflict: the requested condition is supported and the other goes to also_mentions.

Tests: one fixture per reason code; palladium catalyst → reagent_role; "produced in the presence of Y" does not veto; E. coli-only abstract with yeast required → system_mismatch_abstract (conservative) and wrong_system (strict or full text); abstract naming yeast and E. coli with yeast required → supported plus also_mentions; abstract-only paper without the entity → entity_unclear; full text without the entity → entity_absent; G2 criterion uses role patterns as support; direct-negation fixture → condition-level conflicting_evidence.

Phase 6 — Decision Layer (Combiner)

Drafted now, finalized only after the Phase 3.7 probe and Phase 5 behavior are known.

 6.1 Precedence (first matching rule wins).
#	Condition	Result	Reason code	Kind
1	No usable text	insufficient_evidence	no_text	—
2	Role veto fired (entity_role == product)	does_not_qualify	reagent_role	Veto
3	pubType review veto fired	does_not_qualify	review_article	Veto
4a	Entity contradicted (absent from full text)	does_not_qualify	entity_absent	Evidence-based negative
4b	Entity unknown (incl. abstract-only, no match; or demoted by conflict)	insufficient_evidence	entity_unclear / conflicting_evidence	—
5	Activity contradicted (verbatim this-work span; requested activity unsupported)	does_not_qualify	different_activity	Evidence-based negative
6	System required, contradicted, full text (or strict policy)	does_not_qualify	wrong_system	Evidence-based negative
6b	System required, contradicted, abstract-only, conservative policy	insufficient_evidence	system_mismatch_abstract	—
7	Activity unknown	insufficient_evidence	mention_only / activity_unclear / background_citation / possible_review / conflicting_evidence / abstract_only	—
8	System required, unknown	insufficient_evidence	system_unstated / conflicting_evidence	—
9	Unsupported clause present (allow_partial_criterion)	insufficient_evidence	unsupported_criterion_clause	cap on qualify
10	Activity span lacks the entity (evidence only via "the product")	insufficient_evidence	evidence_split	—
11	Entity and activity supported; system supported or not_required	qualifies	none	—
Notes: (a) the unsupported-clause cap (9) is below the negatives, so a clearly wrong-activity paper is rejected rather than abstained; (b) evidence-based negatives never fire when the same condition has supporting evidence; (c) there is no whole-paper conflict rule (see 5.6).
 6.2 Document aggregation: pure max over candidate-sentence condition scores (bonus 0). Named "max" everywhere. The bias is mitigated, not removed, by the candidate cap (4.5) and the conflict rule.
 6.3 Ranking and calibration score = min over supported required conditions. unknown affects the label, not the score.
 6.4 Reason-code registry (app/reasons.py): each code with its label, kind (veto / evidence-based / abstain), meaning, and a test fixture. Required: no_text, abstract_only, mention_only, activity_unclear, entity_unclear, entity_absent, system_unstated, system_mismatch_abstract, conflicting_evidence, evidence_split, unsupported_criterion_clause, different_activity, wrong_system, reagent_role, background_citation, possible_review, review_article. A test fails if any code is emitted without a registry entry.
 6.5 Abstract-only semantics (Issue 3). Abstract-only evidence can fully support qualifies; evidence_source is returned; calibration treats it descriptively (7.2), with no hand-set band downgrade.
 6.6 No learned decision layer in this version.
 6.7 Policy flags (documented in the README): abstract_negative_policy: conservative | strict, allow_partial_criterion: bool.
 6.8 Response schema and versioning (api_version: "v3"; v2 fields deprecated for one release):
json
  {
    "api_version": "v3",
    "classification": "qualifies | does_not_qualify | insufficient_evidence",
    "reason_code": "...",
    "criterion_spec": { "entity_role": "product", "activity": "...", "system": {}, "scope": "this_study", "unparsed_clauses": [] },
    "evidence_source": "abstract_only | fulltext",
    "source_record_sha256": "...",
    "aliases_used": [],
    "retrieved_by": "intent | plain | both",
    "conditions": {
      "entity":   {"status": "...", "score": 0.0, "also_mentions": [], "evidence": [
        {"role": "entity", "raw_text": "...", "display_text": "...", "raw_start": 0, "raw_end": 0,
         "sentence_index": 0, "source": "abstract", "verifiable_alone": true}
      ]},
      "activity": {"status": "...", "score": 0.0, "also_mentions": [], "evidence": []},
      "system":   {"status": "...", "score": 0.0, "also_mentions": [], "evidence": []}
    },
    "evidence_status": "supported | contradicted | insufficient",
    "confidence": {"value": 0.0, "band": "...", "method": "...", "calibration_version": "..."},
    "timings_ms": {}
  }

A schema test fails if any promised field is missing.

Tests: full truth-table test of 6.1 (one case per row plus precedence collisions); no abstention is ever does_not_qualify; an evidence-based negative never fires when the same condition has support; abstract-only paper with no required system can qualify; allow_partial_criterion can reject but never qualify; every non-qualifies result has a registered code.

Phase 7 — Confidence and Ranking (Feasible Scope)
 7.1 Separate concepts. scores.* are raw NLI/feature values; confidence is an approximate, empirically validated estimate that the predicted label is correct. "Posterior probability" appears nowhere.
 7.2 Single primary fit, sized to the data.
Predicted qualifies path (primary): fit P(gold = qualifies | predicted qualifies, score) by LOEO cross-fitting on dev, pooling out-of-fold predictions. Simplest monotone mapping (one-parameter temperature, regularized 2-parameter logistic, or coarse isotonic with ≥3 bins), chosen against v2's sigmoid by out-of-fold Brier.
Predicted does_not_qualify path: fitted only if out-of-fold predicted negatives ≥ N_min (default 30). Otherwise confidence: null, confidence_method: "not_fitted", and the reason code stands in. Never computed as 1 − P.
evidence_source: not a fitting stratum. Reliability is reported descriptively by source; it enters as a binary covariate only if each source has ≥ N_min predicted qualifiers and out-of-fold Brier improves. No hand-set band downgrade.
Report cluster-bootstrap CIs on ECE (3–5 bins) and Brier; state plainly if the fit does not significantly beat v2's.
 7.3 Bands are read from the out-of-fold reliability table (empirical correctness per band), not assumed cut-offs.
 7.4 No fake confidence for abstentions: insufficient_evidence → confidence: null plus its reason_code.
 7.5 Ranking (Issue 20), validated without circularity.
Tier order for non-qualifiers is empirical. On dev (LOEO), measure P(gold = qualifies | predicted insufficient_evidence) vs. P(gold = qualifies | predicted does_not_qualify) with cluster-bootstrap CIs. Rank insufficient_evidence above confident negatives only if the lower CI bound supports it; otherwise rank negatives above abstentions (v2 order). The decision and numbers go in the report.
Primary metrics: binary precision@k and nDCG (qualifies = 1, else 0). A graded nDCG (2/1/0) is a sensitivity row only, flagged as assuming abstentions are more relevant than negatives.
Isolate the reordering effect. Compare three orderings on the same v3 predictions: (a) v3 tiers, (b) v2 tier order, (c) score-only. Plus v2 vs. v3 end to end. Tie-breaks: supported-condition count → pub_year desc → paper_id asc.
 7.6 Recalibration script python -m eval.calibrate writes config/calibration.json (data hash, method, date); the API returns calibration_version.
Phase 8 — Retrieval and Text Access
 8.1 Dual query (1 day, Issue 15). Union the intent-augmented query with plain TITLE_ABS:"{entity}" plus vetted aliases; dedupe; paginate with the Europe PMC cursor; record retrieved_by. Evaluate retrieval separately on a fixed snapshot: retrieval recall of gold-positives and the fraction of clinical/toxicology noise (F6) per query type. v2 and v3 are compared on the same retrieved papers.
 8.2 Quantify abstract-only limits (Issues 3, 16). % of gold-qualifies that are abstract-only; % with open-access full text; how many v3 errors plausibly stem from missing full text; how many predictions change under abstract_negative_policy = strict. Parse full text only if time remains.
 8.3 Deterministic evaluation (Issue 22). Benchmark runs read hashed snapshots; live mode writes timestamped, query-hashed snapshots; responses include aliases_used.
 8.4 Dedup stays; test for preprint→published merge with a missing DOI.
Phase 9 — Model Quality (Document and Quantify)
 9.1 Tag every dev/test error by cause (negation, hedging, numerics, long premises, nomenclature, other), building on the probe.
 9.2 Cheap mitigations only: mean over paraphrases, sentence-boundary truncation, single-sentence verification.
 9.3 Out of scope: fine-tuning, ensembles, larger models; listed as future work with benefit inferred from 9.1.
Phase 10 — Service (Minimal)
 10.1 Profile-driven only. Retrieval-bound → disk cache with TTL and parallel fetching. Inference-bound → ONNX/INT8 only if accuracy cost on dev is measured. No optimization without the Phase 0.5 profile.
 10.2 Out of scope: job queue, /jobs, multi-worker infrastructure. No partial code.
 10.3 Feedback as a curator queue (Issue 23). Multiprocess-safe writes; python -m eval.ingest_feedback validates entries into a candidate file for review; nothing reaches test automatically. README says "curator feedback queue", not "active learning".
 10.4 timings_ms in every response.
Phase 11 — Evaluation, Notebook, Report
 11.1 Protocol. Development: LOEO on dev. After v3-frozen: one pooled dev+test LOEO run (headline) plus test-only (robustness). Reviewer case and proxy reported separately; the proxy is never used for tuning.
 11.2 Required tables (cluster-bootstrap CIs throughout):
Original vs. revised: 3×3 and binary confusion matrices, FP/FN, per-class and macro P/R/F1, abstention and overclaim rates, strict and lenient.
McNemar only with enough discordant pairs.
Out-of-fold reliability, Brier, ECE (7.2).
Coverage-vs-accuracy curve; ranking comparison (7.5, three orderings).
Retrieval-only recall table (8.1).
Policy sensitivity: conservative vs. strict.
 11.3 Ablation: 5 configurations.
Config	Adds
1	v2 baseline
2	+ three-class output
3	+ soft gates and windows (two vetoes only)
4	+ condition decomposition and the hardened combiner
5	+ cross-validated confidence and relevance ranking
CI on every row; overlap with the previous row is reported as "no demonstrated difference". Side table justifying each retained veto, and a second side table for each evidence-based negative (errors with vs. without it).
 11.4 v2-error slice. Slice = papers v2 falsely qualified because a condition was missed. Metric = correction rate, paired with gold-positive recall. Floors set before the ablation runs; a missed floor is reported as "floor not met". Passing is a sanity check, not independent evidence, because the slice shaped the design.
 11.5 Error analysis. 5–8 v2 errors (Pd catalyst, review trap, artemisinic background citation, multi-sentence synthesis, the reviewer case): sentence, v2 verdict, v3 condition breakdown, what changed. Include remaining errors.
 11.6 Reproducible notebook notebooks/evaluation.ipynb: seeds, pinned model revision, offline hashed snapshots, runs v2 and v3, caches predictions, regenerates all tables and plots, ends with assertions on key numbers; "Run All" works offline.
 11.7 Docs. Update ARCHITECTURE.md, KNOWN_ERRORS.md, README.md (grammar, reason codes, policy flags, versioning); regenerate sample_output.json from a real, consistent run.
 11.8 Report: 4–5 pages plus appendix. Body: diagnosis, architecture diagram, changes mapped to issues, headline results, 3–4 error examples, confidence methodology and limits, deliberate deviations D1–D4, quantified limitations. Appendix: full ablation, error tables, constants register and sensitivity, probe results, reliability tables.
Definition of Done
 Every output is one of three labels.
 Every abstain and every negative carries a registered, machine-readable reason_code.
 Only the two vetoes can override supporting evidence; evidence-based negatives always carry a verbatim contradicting span and never fire against supported conditions.
 Every classification has verbatim raw-text evidence from a non-title segment, or evidence_status: "insufficient" with no fabricated text.
 Windows never appear as evidence; evidence is single-sentence spans; the activity span contains the entity or alias; deviation D1 is disclosed.
 Three conditions are evaluated independently; unknown blocks only when the condition is required; conflicts demote one condition, not the paper.
 The criterion parse is shown, overridable, limited to a published grammar, 422 by default for unsupported clauses, and never silently drops a clause.
 Confidence is described as an approximate validated estimate that the predicted label is correct, with CIs, never a posterior; negative-path confidence is null unless enough data.
 The reviewer's failing case (or the labeled proxy) is reproduced on v2 and shown on v3.
 v2 vs. v3 comparison includes FP/FN, P/R/F1, CIs, strict and lenient scoring, abstention and overclaim rates, and the policy sensitivity.
 v2-error slice reports correction rate and gold-positive recall, with pre-registered floors.
 5-row ablation with overlaps stated honestly; veto and evidence-based-negative side tables present.
 Response schema test passes (all promised fields present).
 Notebook reproduces results offline; headline run executed once after v3-frozen.
 All tests pass, with new tests for every phase and a test per reason code.
 Limitations are quantified, not generic.
Status Tracker (update as tests and metrics pass)
Issue	Planned status	Verified by (test / metric)	Verified
1 Recall penalty	Partial	M-recall-strict, M-v2slice, T-unknown-nonblocking, T-no-hard-block-on-nli	☐
2 Single-sentence blindness	Partial	M-gate-recall-span, T-evidence-split, T-window-evidence, T-verifiable-alone	☐
3 Abstract-only	Documented	M-abstract-share, T-evidence-source, M-policy-sensitivity	☐
4 Pseudo-calibration	Partial	M-oof-brier, M-ece-ci	☐
5 Concurrency	Documented	out of scope, noted in report	☐
6 Abstentions as negatives	Planned (Addressed)	T-3class, M-strict-lenient, T-reason-registry	☐
7 Shallow criterion	Planned (Addressed within grammar)	T-parser-grammar, T-unparsed-clause, T-partial-criterion-policy	☐
8 Role hypothesis	Partial	T-role-veto-entity-object, T-agent-role-G2, T-agent-role-activity	☐
9 Ad hoc composite	Partial	T-truth-table, M-constants-sensitivity	☐
10 NLI domain	Documented	M-error-causes, probe results	☐
11 Max bias	Partial (weak)	M-candidate-cap-effect	☐
12 Closed vocabulary	Planned (Addressed)	M-gate-recall-span	☐
13 Alias rules	Partial	T-alias-short-validated, T-related-compound	☐
14 Regex review	Planned (Addressed)	T-review-pubtype, T-we-review-not-vetoed, T-review-score-demotion	☐
15 Retrieval recall	Partial	M-retrieval-recall	☐
16 Section-aware	Partial	T-segment-source	☐
17 Attribution	Partial	T-attribution-scope, T-agentless-failsafe, M-background-errors	☐
18 Offsets	Planned (Addressed)	T-offset-property, T-title-banned, T-schema-fields	☐
19 Confidence on abstain	Partial	T-null-confidence, M-reliability	☐
20 Ranking	Partial	M-p@k, M-nDCG-binary, M-tier-order-empirical, M-reorder-effect	☐
21 Evaluation credibility	Partial	T-leakage, M-cluster-ci, intra-rater κ	☐
22 Reproducibility	Partial	T-offline-identical-runs	☐
23 Feedback loop	Partial	T-ingest-candidate-file	☐
24 Latency	Partial	M-profile, timings_ms	☐
(new) Conflict handling	Planned	T-conflict-condition-level, T-two-organisms-not-conflict	☐
(new) Abstract-only carve-outs	Planned	T-entity-absent-fulltext-only, T-system-mismatch-abstract	☐
Order and Effort (calendar)
Day	Work	Notes
0	Phase 0 baseline; reviewer-case request; Tier 1 labeling finished, hashed, committed	Day 0 is when the relabel clock starts; budget ~3 days for labeling incl. Tier 2 (start Tier 2 on day 1 alongside coding)
1–2	Phase 2 evidence integrity; Tier 2 labeling	
3–4	Phase 3 parser + 3.7 probe	Probe before cut-offs
5–6	Phase 4 candidate generation	
7–9	Phase 5 condition checkers, constants register	Fills the gap; do not touch labels
≥ 8	Blind relabel of ≥30 papers (0.5 day)	Only after ≥7 days since day 0
10	Phase 6 combiner finalized	After probe + Phase 5
11–12	Phase 7 confidence and ranking	
13	Phase 8.1–8.2 dual query and quantification	
14	Phases 9–10 minimal items	
15	git tag v3-frozen	No changes after
16–18	Phase 11 evaluation, notebook, report	

If the reviewer replies late, or labeling runs long, extend the early days; never shorten the 7-day relabel gap. Total ≈ 3.5 weeks of focused work.

Risks and Mitigations
Risk	Mitigation
Abstention inflates precision	Strict/lenient scoring, overclaim rate, macro-F1, coverage curve
NLI-derived negatives reject true positives	Evidence-based negatives never override support, need a verbatim span, abstract-only carve-outs, side table, policy sensitivity
Overfitting to audit papers or test entities	Entity splits, leakage test, frozen tag, single headline run, proxy excluded from tuning
Correlated papers narrow the CIs	Cluster bootstrap by entity
Over-conservatism in either direction	Correction rate paired with gold-positive recall; recall reported strictly
Abstract-only text causes abstentions	unknown ≠ 0; system required only if the user specifies it; entity-absent needs full text
Calibration noise on small data	Single qualifier-path fit, LOEO cross-fitting, few bins, N_min, null when unfittable
Hand-set constants	Register plus sensitivity analysis
Parser misses unusual criteria	Published grammar, 422 default, unparsed_clauses, user override
Offset-map bugs	Per-operation segments, property tests, hashed raw records
Labeling effort underestimated	Two-tier labeling, minimum viable benchmark, report achieved n
Scope creep	Queue, LLM fallback, learned combiner, fine-tuning out of scope