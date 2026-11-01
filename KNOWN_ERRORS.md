# Empirical Classification Error Analysis & Taxonomy (`KNOWN_ERRORS.md`)

**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**Candidate:** Ruhaan Kakar  
**Project:** Mini Literature Classification Engine  

---

## 1. Executive Summary

A core finding of this technical assessment is that generic NLP text-classification and zero-shot NLI models **readily collapse the distinction between "actual activity satisfaction" and "topical entity mention" into generic semantic relatedness**. When queried with scientific literature, standard cross-encoders assign high entailment scores ($\ge 0.95$) to reviews, reagent-use protocols, toxicology case reports, and network pharmacology studies because they are saturated with the entity's vocabulary.

This document details the **seven empirical failure modes** identified during live testing on Europe PMC across three live chemical/biological queries (Caffeine, Palladium(II) acetate, and Artemisinic acid), the root cause of each error, and how the production engine remediates each mode.

---

## 2. Granular Failure Taxonomy & Case Studies

### F1: Lexical Overlap Read as Semantic Entailment (The Review Article Trap)
- **Manifestation:** When testing entity `"caffeine"` with criterion `"The paper describes the chemical synthesis or production of caffeine"`, a literature review titled *"Pharmacological Review of Methylxanthines"* containing the sentence:
  > *"In this review, we synthesize current knowledge on caffeine metabolism and human cardiovascular outcomes."*
  scored **0.987** under unhardened NLI. The word *"synthesize"* (used epistemologically to mean *"summarize/integrate"*) was mapped by the transformer cross-encoder as strong evidence of chemical synthesis.
- **Root Cause:** Zero-shot transformers lack epistemic boundary awareness between epistemological synthesis and laboratory chemical synthesis.
- **Remediation:**
  1. **Review Pre-Filter (`is_review_article`):** Regex filter identifying literature reviews, systematic reviews, and meta-analyses (`\b(review|systematic review|meta-analysis|overview|perspective)\b` or `we synthesize (current )?knowledge`).
  2. **Abstain Path:** Reviews default to `classification: "does_not_qualify"`, `evidence_status: "insufficient"`, `evidence: null`, and low confidence.

---

### F2: Actor / Role Confusion (Reagents & Catalysts Treated as Products)
- **Manifestation:** When testing entity `"palladium(II) acetate"` (`Pd(OAc)2`) with criterion `"The paper describes the synthesis or preparation of palladium(II) acetate as the product"`, a paper titled *"Synthesis of dibenzoxepines"* containing:
  > *"In the presence of palladium(II) acetate and triphenylphosphine, substituted dibenzoxepines were prepared in 78% yield."*
  scored **0.981**. The model recognized a genuine chemical reaction yielding a product, but confused the catalyst with the synthesized target.
- **Root Cause:** Transformers process tokens in self-attention matrices where prepositional heads (`in the presence of`, `catalyzed by`) are frequently overridden by strong co-occurrence associations between the catalyst name and reaction verbs (`prepared in 78% yield`).
- **Remediation:**
  1. **Externalized Role Pattern Engine (`config/role_patterns.json`):** 15 prepositional regex patterns detecting when the target entity sits in reagent, catalyst, or substrate positions (`in the presence of {E}`, `catalyzed by {E}`, `using {E} as catalyst`, `{E}-catalyzed`).
  2. **Negative-Role Hypothesis Comparison:** Evaluates `"{entity} is used as a reagent, catalyst, or starting material"` alongside positive paraphrases. If the role score dominates or a role pattern fires, the paper is disqualified with `guard_reason` logged in the audit trail.

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
  3. **Abstain-Aware Deliberate Ranking Key:**
     Qualifiers (by calibrated probability desc) $\to$ Scored non-qualifiers $\to$ Abstained papers (`insufficient`) $\to$ Papers with no usable text (`no_text`).
     In v2, PMID 42107802 ranks **#1**.

---

### F5: Uncalibrated and Non-Monotone Confidence Bands
- **Manifestation:** In v1, ad-hoc confidence calculations resulted in non-monotonicity: a paper with score `0.8523` was labeled `"medium"` confidence while another with `0.7494` was labeled `"high"`.
- **Root Cause:** Ad-hoc distance formulas combined raw margins with heuristic caps without establishing a unified probability scale.
- **Remediation:**
  1. **Platt Scaling Calibration:** Fitted logistic sigmoid centered at decision threshold $\tau = 0.45$:
     $$P(\text{correct}) = \frac{1}{1 + \exp(-7.0 \cdot (\text{final\_score} - \tau))}$$
  2. **Monotone Qualitative Bands:**
     - High: $P \ge 0.80$
     - Medium: $0.50 \le P < 0.80$
     - Low: $P < 0.50$
  3. Verified by unit test `test_confidence_band_strict_monotonicity`.

---

### F6: Entity-Only Retrieval Starvation
- **Manifestation:** Querying Europe PMC with `TITLE_ABS:"caffeine"` returned the 15 most topically prominent papers—overwhelmingly clinical toxicology, pharmacology, and epidemiology studies. Not a single chemical synthesis paper appeared in the top 25 results, starving the downstream classifier.
- **Root Cause:** Scientific literature search engines rank by citation frequency and topic popularity, not inclusion criteria.
- **Remediation:**
  1. **Intent-Augmented Query Generation:**
     Query derives from criterion intent via `config/intent_map.json`:
     `TITLE_ABS:("caffeine" AND (synthesis OR biosynthesis OR "produced by" OR fermentation OR preparation OR yielded))`
     This directly retrieves synthesis literature without requiring manual query engineering.

---

### F7: Preprint Duplicates & Uncleaned HTML Markup
- **Manifestation:** Europe PMC indexed both the bioRxiv preprint (DOI `10.1101/...`) and the final Nature journal article for the same artemisinic acid study. Additionally, raw abstracts contained embedded HTML tags (`<i>E. coli</i>`, `10<sup>-3</sup>`), causing offset corruption when computing character slice spans.
- **Root Cause:** Incomplete text preprocessing prior to sentence segmentation.
- **Remediation:**
  1. **Pre-Canonicalization Markup Stripping:** `clean_markup_and_unescape()` runs before sentence tokenization, ensuring that character offsets `[start:end]` map 1:1 to the canonical document text.
  2. **DOI & Title Deduplication:** `deduplicate_papers()` compares DOIs and normalized titles, prefers peer-reviewed journal articles over preprints, and records the absorbed preprint ID in `dedup_of`.

---

## 3. Verification Matrix

| Failure Mode | Test File | Verification Check | Status |
|---|---|---|---|
| F1 (Review trap) | `test_model_fixtures.py` | `test_n2_review_article_abstain` | **PASSED** |
| F2 (Role confusion) | `test_model_fixtures.py` | `test_n1_pdoac2_catalyst_role_guard` | **PASSED** |
| F3 (Alias collisions) | `test_remediation_v2_tier1.py` | `test_alias_vetting_denylist` | **PASSED** |
| F4 (Inverted ranking) | `test_model_fixtures.py` | `test_q1_artemisinic_acid_yeast_biosynthesis` | **PASSED** |
| F5 (Uncalibrated bands) | `test_remediation_v2_tier1.py` | `test_confidence_band_strict_monotonicity` | **PASSED** |
| F6 (Retrieval starvation) | `test_live.py` | `test_live_vanillin_europe_pmc` | **PASSED** |
| F7 (Duplicates & markup) | `test_remediation_v2_tier1.py` | `test_deduplication_and_markup_cleaning` | **PASSED** |

