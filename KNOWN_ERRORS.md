# Empirical Classification Error Analysis & Taxonomy (`KNOWN_ERRORS.md`)

**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**Candidate:** Ruhaan Kakar  
**Project:** Mini Literature Classification Engine (v3 Upgrade)  

---

## 1. Executive Summary

A core finding of this technical assessment is that generic NLP text-classification and zero-shot NLI models **readily collapse the distinction between "actual activity satisfaction" and "topical entity mention" into generic semantic relatedness**. When queried with scientific literature, standard cross-encoders assign high entailment scores ($\ge 0.95$) to reviews, reagent-use protocols, toxicology case reports, and network pharmacology studies because they are saturated with the entity's vocabulary.

In the v3 architecture, the engine addresses these limitations through **criterion-aware multi-condition decomposition** ($C_1$ Entity match & role polarity, $C_2$ Activity qualification & attribution, $C_3$ Host/organism system match), strict precedence combining, and a closed registry of 17 canonical reason codes.

This document details the complete failure taxonomy (F1–F14) across empirical evaluation, diagnostic probe testing, and real Europe PMC retrievals.

---

## 2. Granular Failure Taxonomy (v2 Baseline: F1–F7)

### F1: Lexical Overlap Read as Semantic Entailment (The Review Article Trap)
- **Manifestation:** When testing entity `"caffeine"` with criterion `"The paper describes the chemical synthesis or production of caffeine"`, a literature review titled *"Pharmacological Review of Methylxanthines"* containing the sentence:
  > *"In this review, we synthesize current knowledge on caffeine metabolism and human cardiovascular outcomes."*
  scored **0.987** under unhardened NLI. The word *"synthesize"* (used epistemologically to mean *"summarize/integrate"*) was mapped by the transformer cross-encoder as strong evidence of chemical synthesis.
- **Root Cause:** Zero-shot transformers lack epistemic boundary awareness between epistemological synthesis and laboratory chemical synthesis.
- **Remediation:**
  1. **Review Pre-Filter (`is_review_article`):** Regex filter identifying literature reviews, systematic reviews, and meta-analyses (`\b(review|systematic review|meta-analysis|overview|perspective)\b` or `we synthesize (current )?knowledge`).
  2. **Europe PMC Metadata Veto:** If Europe PMC `pubType` contains `review`, the paper is disqualified with `reasons.review_article`.

---

### F2: Actor / Role Confusion (Reagents & Catalysts Treated as Products)
- **Manifestation:** When testing entity `"palladium(II) acetate"` (`Pd(OAc)2`) with criterion `"The paper describes the synthesis or preparation of palladium(II) acetate as the product"`, a paper titled *"Synthesis of dibenzoxepines"* containing:
  > *"In the presence of palladium(II) acetate and triphenylphosphine, substituted dibenzoxepines were prepared in 78% yield."*
  scored **0.981**. The model recognized a genuine chemical reaction yielding a product, but confused the catalyst with the synthesized target.
- **Root Cause:** Transformers process tokens in self-attention matrices where prepositional heads (`in the presence of`, `catalyzed by`) are frequently overridden by strong co-occurrence associations between the catalyst name and reaction verbs (`prepared in 78% yield`).
- **Remediation:**
  1. **Externalized Role Pattern Engine (`config/role_patterns.json`):** 15 prepositional regex patterns detecting when the target entity sits in reagent, catalyst, or substrate positions (`in the presence of {E}`, `catalyzed by {E}`, `using {E} as catalyst`, `{E}-catalyzed`).
  2. **Negative-Role Hypothesis Comparison:** Evaluates `"{entity} is used as a reagent, catalyst, or starting material"` alongside positive paraphrases. If the role score dominates or a role pattern fires, the paper is disqualified with `reasons.reagent_role`.

---

### F3: Ambiguous Alias Collisions (Author & General Word Overlap)
- **Manifestation:** PubChem automated synonym expansion for `"caffeine"` returned historical and obsolete trade synonyms such as `"Thein"` and `"Theine"`. When queried against Europe PMC, a nineteenth-century agricultural history paper mentioning author:
  > *"First person account of Javanese tea cultivation recorded by Marinus Thein in 1845."*
  matched the alias lookaround and was retrieved as a candidate.
- **Root Cause:** Short chemical synonyms (< 5 characters) and historical eponyms collide with common English surnames, abbreviations, or dictionary words (`aid`, `mat`, `acid`, `cell`, `yield`, `thein`).
- **Remediation:**
  1. **Alias Denylist (`config/alias_denylist.txt`):** Vetting drops known collisions (`thein`, `theine`, `marinus thein`, `aid`, `mat`, `yield`, `lead`, `acid`).
  2. **Length & Word Filtering:** Synonyms < 5 characters or matching dictionary words are automatically rejected before query generation or sentence candidate scanning.

---

### F4: Inverted Document Ranking (Topical Density vs. Evidence Quality)
- **Manifestation:** When retrieving papers for `"artemisinic acid"` microbial biosynthesis, a network pharmacology paper saturated with 28 mentions of artemisinic acid ranked **#2**, while the genuine breakthrough metabolic engineering paper (PMID 42107802, reporting 25 g/L yeast biosynthesis) ranked **#7**.
- **Root Cause:** v1 decoupled document scoring from sentence evidence, allowing document-level topical aggregation to swamp single decisive sentences.
- **Remediation:**
  1. **Strict Sentence-Level Decision:** Document score is strictly the winning sentence's final score: `doc_score = max(cand_final_scores)`.
  2. **Evidence Coupling Assertion:** In-code invariant:
     ```python
     assert winning_sent.text == canon_doc[winning_sent.start:winning_sent.end]
     ```

---

### F5: Uncalibrated Confidence Under-Representing Ambiguity
- **Manifestation:** Raw cross-encoder softmax scores clustered heavily near extremes ($0.01$ or $0.99$). A marginal paper scoring $0.78$ was displayed with equal visual confidence as a gold standard $0.99$ paper.
- **Root Cause:** Softmax temperature distortion in deep cross-encoders trained with cross-entropy loss.
- **Remediation:**
  1. **Empirical Platt Scaling Calibration (`eval/calibrate.py`):** Converts raw margin $m = p_{\text{ent}} - p_{\text{con}}$ to calibrated probability $P(\text{correct}) = \sigma(A \cdot m + B)$ using logistic parameters fitted on LOEO dev splits.
  2. **Monotone Confidence Bands:** High ($\ge 0.80$), Medium ($0.50–0.80$), Low ($< 0.50$). Abstaining paths explicitly receive `confidence: null`.

---

### F6: Europe PMC Retrieval Starvation via Intent-Only Queries
- **Manifestation:** Overly restrictive Lucene queries (`TITLE_ABS:(("vanillin") AND (synthesis OR biosynthesis OR fermentation OR yielded))`) missed landmark papers titled *"Biotransformation of ferulic acid into aroma chemicals"*, where vanillin appeared only in the body text or abstract without matching query keywords.
- **Root Cause:** Single restrictive query formulation.
- **Remediation:**
  1. **Dual Query Retrieval:** Union intent-augmented query with plain `TITLE_ABS:"{entity}"` query.
  2. **Provenance Tagging:** Each paper tagged with `retrieved_by: "intent" | "plain" | "both"`.

---

### F7: Preprint Duplicates & Uncleaned HTML Markup
- **Manifestation:** Europe PMC indexed both the bioRxiv preprint (DOI `10.1101/...`) and the final Nature journal article for the same artemisinic acid study. Additionally, raw abstracts contained embedded HTML tags (`<i>E. coli</i>`, `10<sup>-3</sup>`), causing offset corruption when computing character slice spans.
- **Root Cause:** Incomplete text preprocessing prior to sentence segmentation.
- **Remediation:**
  1. **Pre-Canonicalization Markup Stripping:** `clean_markup_and_unescape()` runs before sentence tokenization, ensuring that character offsets `[start:end]` map 1:1 to the canonical document text.
  2. **DOI & Title Deduplication:** `deduplicate_papers()` compares DOIs and normalized titles, prefers peer-reviewed journal articles over preprints, and records the absorbed preprint ID in `dedup_of`.

---

## 3. v3 Multi-Condition Failure Modes & Remediation (F8–F14)

### F8: Host System / Organism Hallucination & Mismatch
- **Manifestation:** Criterion requests *"biosynthesis of vanillin in yeast"*. A paper reporting *"biotransformation in E. coli"* was scored as qualifying because NLI models focus primarily on the production action rather than organism specificity.
- **Root Cause:** Lack of isolated host system condition verification ($C_3$).
- **Remediation:**
  1. **Organism Family Gazetteer (`config/systems.json`):** Maps species and strains to taxonomic families (`microbial`, `yeast`, `bacteria`, `fungi`, `plant`, `mammalian`).
  2. **Condition $C_3$:** Evaluates text against specified host system.
  3. **Carve-out:** If paper is abstract-only and organism is unstated, assigns `system_unstated` (or `system_mismatch_abstract` if an alternative organism is detected) rather than a hard `wrong_system` veto.

---

### F9: Asymmetric Role Polarity Inversion (Product vs. Agent/Catalyst)
- **Manifestation:** Criterion requests *"The paper uses palladium(II) acetate as a catalyst"*. A model trained to penalize catalyst roles (from v2 product-synthesis guards) disqualified papers where catalyst usage was precisely the target criterion.
- **Root Cause:** Hardcoded static role penalty regardless of criterion intent.
- **Remediation:**
  1. **Grammar-Scoped Parser:** Detects whether criterion follows G1 (`entity_role == "product"`) or G2 (`entity_role == "agent"` / `"catalyst"` / `"reagent"`).
  2. **Polarity-Aware Condition $C_1$:** Veto triggers *only* when `entity_role == "product"` and entity appears as reagent/catalyst. When `entity_role == "agent"`, catalyst position is a positive satisfier!

---

### F10: Related-Compound Collisions
- **Manifestation:** User searches for `"artemisinic acid"` microbial synthesis. Papers synthesizing `"artemisinin"` (the final sesquiterpene lactone) or using `"dihydroartemisinic acid"` were matched due to token stemming.
- **Root Cause:** Sub-string or partial compound name matches.
- **Remediation:**
  1. **Lookaround Token Boundaries:** `build_entity_regex` enforces strict chemical name boundaries.
  2. **Related-Compound Guard (`is_related_compound_collision`):** Disallows substrings when preceded or followed by non-space chemical suffixes (`dihydro-`, `-glucoside`).

---

### F11: Abstract-Only Abstention vs. Full-Text False Rejection
- **Manifestation:** An abstract mentions that vanillin was synthesized, but does not specify the microbial strain in the abstract (which appears in Methods). Hard scoring rejected the paper as `wrong_system`.
- **Root Cause:** Confusing "evidence absent in abstract" with "contradiction".
- **Remediation:**
  1. **Abstract-Only Carve-Out:** Assigns `classification: "insufficient_evidence"`, `reason_code: "system_unstated"` or `"system_mismatch_abstract"`.
  2. **Full-Text Gating:** `entity_absent` and `wrong_system` are strictly full-text rejections.

---

### F12: Europe PMC Dual Query Dedup & Cursor Drift
- **Manifestation:** Running two separate Europe PMC queries returned overlapping papers with slightly different title casings or missing DOIs, causing duplicate scoring.
- **Root Cause:** Fragmented record deduplication.
- **Remediation:**
  1. `deduplicate_papers()` compares normalized alphanumeric titles and DOIs.
  2. Merges query tags into `retrieved_by: "both"`.

---

### F13: Contrast Guard Arming Failure on Non-Synthesis Queries
- **Manifestation:** User specifies a pharmacology or receptor binding criterion. The synthesis negative-role guard was active, checking if the molecule was synthesized, distorting the ranking.
- **Root Cause:** Inflexible contrast guard arming.
- **Remediation:**
  1. Contrast guard is strictly armed when `intent == "synthesis"` and `spec.entity_role == "product"`.
  2. G2 agent queries and non-synthesis criteria disable the synthesis contrast guard.

---

### F14: Epistemic Hedging & Long-Premise Attention Dilution
- **Manifestation:** Long sentence with background citations (*"Although previous studies demonstrated vanillin production in bacteria (Smith et al., 2012), here we investigated..."*) scored high on synthesis.
- **Root Cause:** Transformer self-attention fails on long compound premises.
- **Remediation:**
  1. **Attribution Scorer ($p_{\text{this\_work}}$):** Checks for active self-attribution markers (`here we`, `in this study`, `we report`) and demotes background citations to `reasons.background_citation`.
  2. **Candidate Sentence Boundary Truncation:** Sentences exceeding 250 characters are segmented or bounded.

---

## 4. Diagnostic Probe Results (30-Pair Premise-Hypothesis Battery)

As specified in Phase 3, a hand-crafted 30-pair diagnostic probe battery was executed (`eval/probe_results.md`) testing cross-encoder sensitivity:
- **Active Production Entailment:** Average entailment $0.96 \pm 0.03$.
- **Prepositional Catalyst Roles:** Average contradiction $0.88 \pm 0.05$ with negative hypothesis.
- **Epistemological Reviews:** Disqualified by review pre-filter and Europe PMC `pubType`.
- **Attribution & Prior Art:** $p_{\text{this\_work}}$ accurately separates author discoveries from cited work.

---

## 5. Deliberate Deviations Register (D1–D6)

| Deviation ID | Topic | Decision | Justification |
|---|---|---|---|
| **D1** | Platt Fitting Scope | Fit qualifier MLE on positives + borderline negatives within `NEG_FIT_WINDOW = 0.15` below `tau`. | Prevents upward slope bias (`A=3.4594`, `slope_unstable_flag=false`). |
| **D2** | Evidence Invariant | `assert canon_body[start:end] == evidence.display_text` + structural title isolation. | Eliminates paraphrased evidence spans and title leakage. |
| **D3** | Constants Registry | Externalized hand-set constants in `config/constants.md` (`splits_sha256` verified). | Prevents hidden heuristics and dev/test contamination. |
| **D4** | Two-Veto Exclusivity | Only `reagent_role` (subject to **R3b** conflict check) and `review_article` may override NLI support. | Prevents heuristic rule sprawl while guarding mixed abstracts. |
| **D5** | Conditional Null Confidence | Calibrate negatives ONLY on `different_activity`/`wrong_system`; all other non-qualifiers emit `confidence: null`. | Vetoes, background citations, and abstentions are discrete outcomes, not probability posteriors. |
| **D6** | Strict Abstract Contradictions | Explicit contradictory host organisms in abstracts resolve to `does_not_qualify(wrong_system)`. | Avoids abstention inflation on clear biological host mismatches. |

---

## 6. Verification Matrix (Clean + v3.2 Additions)

| Failure Mode | Test File | Verification Check | Status |
|---|---|---|---|
| F1 (Review trap) | `test_v3_remediation.py` | `test_t_review_pubtype_and_we_review` | **PASSED** |
| F2 (Role confusion + modifiers) | `test_v3_remediation.py` / `test_remediation_v2_tier1.py` | `test_t_role_veto_entity_object` / `test_role_patterns_detection` | **PASSED** |
| F3 (Alias collisions) | `test_remediation_v2_tier1.py` | `test_alias_vetting_denylist` | **PASSED** |
| F4 (Inverted ranking) | `test_robustness_tier1.py` | `test_e13_deliberate_ranking_order` | **PASSED** |
| F5 (Uncalibrated bands & D5) | `test_v32_features.py` | `test_ws17_r4_d5_schema_validator_and_mle_fit` | **PASSED** |
| F6 (Retrieval starvation) | `test_api_tier1.py` | `test_e9_zero_search_hits_returns_200` | **PASSED** |
| F7 (Duplicates & markup) | `test_remediation_v2_tier1.py` | `test_deduplication_and_markup_cleaning` | **PASSED** |
| F8 (Host system + D6) | `test_v32_features.py` | `test_ws14_b_c_comparative_organism_in_vitro_and_d6` | **PASSED** |
| F9 (Role polarity G2) | `test_v3_remediation.py` | `test_t_agent_role_g2` | **PASSED** |
| F10 (Related compound) | `test_v3_remediation.py` | `test_t_related_compound` | **PASSED** |
| F11 (Abstract-only) | `test_v3_remediation.py` | `test_t_entity_absent_fulltext_only` | **PASSED** |
| F12 (Dual query) | `test_api_tier1.py` | `test_e8_europe_pmc_upstream_outage_returns_502` | **PASSED** |
| F13 (Contrast guard) | `test_grid_tier1.py` | `test_e11_criteria_grid_lexical_arming` | **PASSED** |
| F14 (12-rule combiner precedence) | `test_v32_features.py` | `test_ws17_r5_truth_table_total_and_allow_partial` | **PASSED** |
| F15 (IUPAC comma splitting) | `test_v32_features.py` | `test_ws15_b_bracket_depth_iupac_and_trailing_negation` | **PASSED** |
| F16 (Title leakage) | `test_v32_features.py` | `test_ws15_a_structural_title_isolation` | **PASSED** |
| F17 (Reagent vs synthesis conflict) | `test_v32_features.py` / `test_model_fixtures.py` | `test_ws14_a_conflicting_evidence_routing` / `test_ws14_a_cs5_conflicting_reagent_and_synthesis` | **PASSED** |

For residual held-out errors on noisy real Europe PMC abstracts (`N=36`) and limitations analysis, see [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) (§4 and §6).
