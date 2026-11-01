# Mini Literature Classification Engine

**Candidate:** Ruhaan Kakar  
**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**Task:** Mini Literature Classification Engine (Technical Assessment)  
**Repository:** `loopworm` / `literature-classifier`

---

## Executive Summary & Assessment Checklist

This project delivers a modular, production-grade semantic literature mining prototype that retrieves scientific publications from the **Europe PMC REST API**, extracts and tokenizes candidate sentences using biomedical NLTK Punkt, performs batched Natural Language Inference (NLI) using a local Cross-Encoder (`cross-encoder/nli-deberta-v3-small`), and assigns an evidence-traceable binary classification (`qualifies` / `does_not_qualify`) accompanied by symmetric confidence scores, verbatim supporting evidence, and an active learning feedback loop.

### Key Hardening Highlights:
- **Starting Material Separation (P0 Fix):** Eliminated the `xfail` on reagent/starting-material distinction. DeBERTa's trailing negative clause penalty was resolved by affirmative multi-contrast hypotheses (`"{entity} is a starting material or reactant in this reaction"`), scoring $`P(\text{ent}) = 0.977`$ on reactant sentences and enabling the contrast guard to fire reliably.
- **Paraphrase Ensemble:** Median aggregation across 3 canonical deterministic paraphrases narrows zero-shot phrasing sensitivity from 40–70% to 55–75% holdout precision.
- **Honest Statistical Reporting:** Leads with Holdout Precision and 95% non-parametric bootstrap confidence intervals ($`n=2000`$). Documents threshold saturation via empirical margin value histograms.
- **No Invented Evidence Guarantee:** Enforced at runtime via `_assert_verbatim()`, raising `EvidenceIntegrityError` if any extracted evidence sentence is not an exact substring of source text.
- **Non-Blocking Architecture:** FastAPI route handlers execute on AnyIO worker threadpools, preventing GIL and event-loop starvation during PyTorch inference.
- **Resilient Retrieval:** Exponential backoff retry loop on Europe PMC 429 / 5xx responses with open-access full-text XML extraction for PMC articles.

| Requirement | Implementation Detail | Section Link |
| :--- | :--- | :--- |
| **1. Literature Retrieval** | Europe PMC REST API (`TITLE_ABS` queries, boolean synonyms, retry backoff, open-access XML fallback) | [Pipeline Walkthrough](#step-by-step-pipeline-walkthrough) |
| **2. Semantic Classification** | NLI decision margin $`m = P(\text{entail}) - P(\text{contradict})`$ against criterion with affirmative contrast guard | [Methodology](#methodology--mathematical-formulations) |
| **3. Evidence Extraction** | Verbatim sentence extraction verified by runtime assertion; fallback to `insufficient_text` | [Evidence Rules](#evidence-rule-table--verbatim-guarantee) |
| **4. Ranking & Confidence** | Coherent ranking via $`\text{qualify-score} = (m+1)/2`$; symmetric boundary distance; runner-up margin gap pooling | [Scoring & Confidence](#ranking--confidence-scoring) |
| **5. Structured JSON Output** | Validated Pydantic schema with PMID, DOI, PMCID, evidence source, classification, and confidence | [API Specification](#api-specification--endpoints) |
| **6. FastAPI Implementation** | Non-blocking endpoints (`POST /classify`, `GET /health`, `POST /feedback`) with lifespan warmup | [API Documentation](#api-specification--endpoints) |
| **7. Technical Constraints** | 100% Python, free data (Europe PMC), zero paid LLMs (local DeBERTa-v3 on CPU/CUDA) | [Architecture](#architecture--design-decisions) |
| **8. Real Data Evaluation** | 40-paper labeled benchmark across 2 entities (`psilocybin` tuned, `vanillin` holdout) with 95% bootstrap CIs | [Empirical Evaluation](#empirical-evaluation--statistical-rigor) |
| **9. Real Case Studies & Errors** | Analysis of qualifiers, mere mentions, reagent fallacy, and margin saturation | [Case Studies & Errors](#case-studies--categorized-error-analysis) |
| **10. User Feedback Loop** | Active learning queue, JSONL persistence, logistic calibration, and domain fine-tuning roadmap | [Continuous Improvement](#user-feedback--continuous-improvement-plan) |

---

## Table of Contents
1. [Quick Navigation & Commands](#quick-navigation--commands)
2. [Architecture & Design Decisions](#architecture--design-decisions)
3. [Methodology & Mathematical Formulations](#methodology--mathematical-formulations)
   - [NLI Margin & Decision Rule](#nli-margin--decision-rule)
   - [Affirmative Contrast Guard Hypotheses](#affirmative-contrast-guard-hypotheses)
   - [Ranking & Confidence Scoring](#ranking--confidence-scoring)
   - [Evidence Rule Table & Verbatim Guarantee](#evidence-rule-table--verbatim-guarantee)
4. [Step-by-Step Pipeline Walkthrough](#step-by-step-pipeline-walkthrough)
5. [API Specification & Endpoints](#api-specification--endpoints)
6. [Testing Suite](#testing-suite)
7. [Empirical Evaluation & Statistical Rigor](#empirical-evaluation--statistical-rigor)
   - [Holdout Evaluation on Entity B (Vanillin) with 95% Bootstrap CIs](#holdout-evaluation-on-entity-b-vanillin-with-95-bootstrap-cis)
   - [Threshold Saturation & Margin Value Histogram](#threshold-saturation--margin-value-histogram)
   - [Criterion Phrasing Sensitivity & Ensemble Impact](#criterion-phrasing-sensitivity--ensemble-impact)
   - [Tuning on Entity A (Psilocybin)](#tuning-on-entity-a-psilocybin)
8. [Case Studies & Categorized Error Analysis](#case-studies--categorized-error-analysis)
9. [User Feedback & Continuous Improvement Plan](#user-feedback--continuous-improvement-plan)
10. [Assumptions, Limitations & Future Work](#assumptions-limitations--future-work)

---

## Quick Navigation & Commands

You can run and evaluate all aspects of the codebase from the project root using standard Python commands.

### 1. Environment Setup
```bash
# Clone or navigate into repository
cd loopworm/literature-classifier

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Sample Pipeline (Live Europe PMC Retrieval)
Fetches publications for psilocybin from Europe PMC, runs NLI classification, prints a summary table, and writes `sample_output.json`:
```bash
# From workspace root:
python literature-classifier/scripts/run_sample.py

# Or from inside literature-classifier/:
python scripts/run_sample.py
```

### 3. Start the FastAPI Web Service
Starts the REST API server with automatic lifespan warm-up (pre-loads DeBERTa weights and NLTK Punkt tokenizer into memory for sub-second inference):
```bash
# From workspace root:
uvicorn app.main:app --app-dir literature-classifier --host 127.0.0.1 --port 8000

# Or from inside literature-classifier/:
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: `curl http://127.0.0.1:8000/health`

### 4. Execute the Test Suite
```bash
# Run all unit and API tests (fast mock scorer, 20 tests):
python -m pytest literature-classifier/tests -m "not slow" -v

# Run entire test suite including real DeBERTa neural integration tests (24 tests, 100% green):
python -m pytest literature-classifier/tests -v
```

### 5. Run Benchmark Evaluation & Tuning
```bash
# Run grid search threshold tuning on Entity A (Psilocybin):
python literature-classifier/eval/tune_threshold.py

# Run holdout evaluation on Entity B (Vanillin) with 95% bootstrap CIs and histogram:
python literature-classifier/eval/evaluate.py
```

---

## Architecture & Design Decisions

| Decision | Selected Choice | Technical Rationale |
| :--- | :--- | :--- |
| **Decision Signal** | NLI Margin $`m = P(\text{entail}) - P(\text{contradict}) \in [-1, 1]`$ | Raw entailment $`P(\text{entail})`$ is inflated by scientific background statements; margin $`m`$ strictly penalizes contradiction and requires positive evidence to dominate. |
| **Decision Rule** | `qualifies` if $`m_{\text{best}} \ge \tau`$ and contrast guard silent | Clear geometric decision boundary. Empirically tuned to $`\tau^* = 0.65`$, $`\delta^* = 0.05`$ on Entity A. |
| **Affirmative Contrast Guard** | 3 affirmative hypotheses without trailing negative clauses | DeBERTa assigns severe contradiction penalties ($`P_{\text{con}} \approx 0.995`$) to sentences ending with `", not produced"`. Affirmative templates (`"{entity} is a starting material or reactant in this reaction"`) score $`P_{\text{ent}} = 0.977`$, resolving the reagent fallacy. |
| **Paraphrase Ensemble** | Median margin across 3 canonical deterministic phrasings | Neutralizes syntactic sensitivity in zero-shot cross-encoders, narrowing precision variance between passive and agentive queries. |
| **Ranking Score** | $`\text{qualify-score} = \frac{m_{\text{best}} + 1.0}{2.0} \in [0, 1]`$ | Monotonically derived from decision margin. No heuristic bonuses in the decision signal; title matches and mention counts only break ties. |
| **Symmetric Confidence** | Distance from boundary: $`[0.5, 1.0]`$ per side | A paper with $`m = -0.90`$ is a confident negative. Distance from $`\tau`$ ensures symmetric calibration for both classes. |
| **Runner-Up Margin Pooling** | Band capped at `medium` if $`m_{\text{best}} - m_{\text{runner-up}} \lt 0.10`$ | Distinguishes clear-cut evidence from papers where multiple sentences compete closely. |
| **Verbatim Assertion** | Runtime check `_assert_verbatim(evidence, pub)` | Guarantees that no evidence is ever hallucinated or modified. Raises `EvidenceIntegrityError` if breached. |
| **Non-Blocking Handlers** | FastAPI synchronous `def` endpoints | FastAPI dispatches synchronous `def` functions to AnyIO worker threadpools, preventing PyTorch CPU/CUDA execution from blocking the async event loop. |
| **Full-Text Fallback** | Europe PMC XML API (`/{PMCID}/fullTextXML`) | Enables open-access papers with PMCID to search experimental methods when abstracts omit reaction details. |
| **Zero Paid LLMs** | Local `cross-encoder/nli-deberta-v3-small` | 100% reproducible, privacy-preserving, runs locally on CPU or CUDA without external API dependencies. |

---

## Methodology & Mathematical Formulations

### NLI Margin & Decision Rule

For a candidate sentence $`s_i`$ and the user criterion $`H`$, the Cross-Encoder outputs softmax probabilities over three classes: $`(P(E), P(N), P(C))`$.

The **decision margin** $`m_i`$ for candidate sentence $`i`$ is:

```math
m_i = P_i(\text{entailment}) - P_i(\text{contradiction}) \in [-1.0, 1.0]
```

Across the ensemble of $`K`$ criterion paraphrases, the sentence margin is the median:

```math
m_i = \operatorname{median}_{k=1 \dots K} \left( P_{i,k}(\text{entailment}) - P_{i,k}(\text{contradiction}) \right)
```

The best candidate sentence for the publication is selected via:

```math
i^* = \arg\max_{i} m_i, \qquad m_{\text{best}} = m_{i^*}, \qquad P_{\text{ent,best}} = P_{i^*}(\text{entailment})
```

The classification decision is:

```math
\text{classification} =
\begin{cases}
\text{qualifies}, & \text{if } m_{\text{best}} \ge \tau \text{ and the contrast guard is silent} \\
\text{does-not-qualify}, & \text{otherwise}
\end{cases}
```

---

### Affirmative Contrast Guard Hypotheses

To distinguish papers where the target entity is synthesized from those where it is consumed as a reactant or administered as a subject, the engine evaluates affirmative contrast hypotheses:

1. `"{entity} is a starting material or reactant in this reaction."`
2. `"{entity} was used as a reagent, solvent, or analytical standard in this study."`
3. `"{entity} was administered, consumed, or given to subjects or participants in this study."`

The contrast entailment for candidate sentence $`i^*`$ is:

```math
c_{\text{best}} = \max_{j} P_{i^*, j}(\text{contrast entailment})
```

The contrast guard fires if and only if:

```math
\text{contrast-guard-fired} =
\begin{cases}
\text{True}, & \text{if } c_{\text{best}} \gt P_{\text{ent,best}} + \delta \\
\text{False}, & \text{otherwise}
\end{cases}
```

If the criterion does not relate to synthesis (e.g. *"Compound showed antifungal activity"*), the contrast guard is marked `not_applicable` and bypassed.

---

### Ranking & Confidence Scoring

**1. Ranking Score**

```math
\text{qualify-score} = \frac{m_{\text{best}} + 1.0}{2.0} \in [0.0, 1.0]
```

For papers with `insufficient_text`, the qualify score is `None` (sorted strictly last).

**2. Symmetric Confidence**

```math
\text{confidence} =
\begin{cases}
0.5 + 0.5 \times \dfrac{m_{\text{best}} - \tau}{1.0 - \tau}, & \text{if qualifies} \\[2ex]
0.65, & \text{if does-not-qualify (contrast guard flipped)} \\[1ex]
0.5 + 0.5 \times \dfrac{\tau - m_{\text{best}}}{\tau + 1.0}, & \text{if does-not-qualify (normal)}
\end{cases}
```

**3. Confidence Bands**

- `high`: $`\text{conf} \ge 0.80`$ and $`m_{\text{best}} - m_{\text{runner-up}} \ge 0.10`$
- `medium`: $`0.60 \le \text{conf} \lt 0.80`$ (or high confidence with close runner-up sentence ambiguity)
- `low`: $`\text{conf} \lt 0.60`$
- `insufficient`: No usable text mentioning entity

---

### Evidence Rule Table & Verbatim Guarantee

| Classification | Condition | Evidence Selected | Status Flag |
| :--- | :--- | :--- | :--- |
| `qualifies` | $`m_{\text{best}} \ge \tau`$ and contrast guard silent | Candidate sentence $`i^*`$ | `entailing_sentence` |
| `does_not_qualify` | $`m_{\text{best}} \ge \tau`$ but contrast guard fired | Candidate with highest contrast entailment | `contrast_mention` |
| `does_not_qualify` | $`m_{\text{best}} \lt \tau`$ and usable text exists | Candidate with highest contrast or best margin | `strongest_mention` |
| `does_not_qualify` | No abstract and title lacks entity | `None` | `insufficient_text` |

**Verbatim Guarantee:** All extracted evidence strings are validated against the source text before returning. If any character sequence does not appear verbatim in the paper's title, abstract, or fulltext sections, an `EvidenceIntegrityError` is raised.

---

## Step-by-Step Pipeline Walkthrough

```
User Query: entity="psilocybin", criterion="Psilocybin was synthesized..."
     │
     ▼
[1. Europe PMC Retrieval] ──► TITLE_ABS query with boolean synonyms + retry backoff
     │
     ▼
[2. Text Cleaning & Tokenization] ──► Strip HTML, NLTK Punkt (biomedical abbrevs), lookaround regex
     │
     ▼
[3. Candidate Sentence Extraction] ──► Pre-ranked title + abstract + open-access XML sections
     │
     ▼
[4. Batched NLI Inference] ──► DeBERTa-v3 scores 3 criterion paraphrases + 3 contrast hypotheses
     │
     ▼
[5. Decision & Contrast Guard] ──► Median criterion margin vs max affirmative contrast entailment
     │
     ▼
[6. Verbatim Assertion & Ranking] ──► Verify substring integrity; sort by qualify_score descending
```

---

## API Specification & Endpoints

### 1. Health Check (`GET /health`)
```bash
curl http://127.0.0.1:8000/health
```
```json
{
  "status": "healthy",
  "model_name": "cross-encoder/nli-deberta-v3-small",
  "model_revision": "fa2804872c3b4bd748f38c0185cc85775361e735",
  "device": "cuda"
}
```

### 2. Classify Publications (`POST /classify`)
```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "entity": "psilocybin",
    "criterion": "The authors synthesized or biosynthesized psilocybin in this study.",
    "synonyms": ["psilocybine", "4-phosphoryloxy-DMT"],
    "max_results": 5,
    "include_fulltext": false
  }'
```
```json
{
  "query_used": "TITLE_ABS:(\"psilocybin\" OR \"psilocybine\" OR \"4-phosphoryloxy-DMT\")",
  "normalised_criterion": "The authors synthesized or biosynthesized psilocybin in this study.",
  "threshold": 0.65,
  "contrast_delta": 0.05,
  "hypotheses_used": [
    "The authors synthesized or biosynthesized psilocybin in this study.",
    "In this study, the authors carried out the work described as: The authors synthesized or biosynthesized psilocybin in this study.",
    "This paper reports as its direct experimental result: The authors synthesized or biosynthesized psilocybin in this study.",
    "Psilocybin is a starting material or reactant in this reaction.",
    "Psilocybin was used as a reagent, solvent, or analytical standard in this study.",
    "Psilocybin was administered, consumed, or given to subjects or participants in this study."
  ],
  "contrast_guard_active": "active",
  "total_retrieved": 5,
  "results": [
    {
      "title": "Magic pathways: the genes and enzymes of psilocybin biosynthesis.",
      "pmid": "42937360",
      "doi": "10.1128/mmbr.00117-25",
      "pmcid": null,
      "classification": "qualifies",
      "evidence": "In this article, we focus on the metabolic processes leading to psilocybin in the fungal cell and present the biosynthetic enzymes, their structures, and the reactions catalyzed.",
      "evidence_status": "entailing_sentence",
      "evidence_source": "abstract",
      "qualify_score": 0.993,
      "confidence": 0.9797,
      "confidence_band": "high"
    }
  ]
}
```

### 3. Feedback Endpoint (`POST /feedback`)
Persists curator corrections into `feedback.jsonl` for continuous model improvement.

---

## Testing Suite

The test suite contains **24 tests (100% passing)** across unit, API, and neural integration tiers:

```bash
python -m pytest literature-classifier/tests -v
```

```
literature-classifier/tests/test_api.py::test_health_endpoint PASSED
literature-classifier/tests/test_api.py::test_classify_success PASSED
literature-classifier/tests/test_api.py::test_classify_zero_results PASSED
literature-classifier/tests/test_api.py::test_classify_retrieval_failure_502 PASSED
literature-classifier/tests/test_api.py::test_classify_validation_error PASSED
literature-classifier/tests/test_api.py::test_feedback_endpoint PASSED
literature-classifier/tests/test_api.py::test_retrieval_retry_on_429 PASSED
literature-classifier/tests/test_api.py::test_fulltext_xml_extraction PASSED
literature-classifier/tests/test_integration.py::test_real_model_synthesis_qualifies PASSED
literature-classifier/tests/test_integration.py::test_real_model_administered_does_not_qualify PASSED
literature-classifier/tests/test_integration.py::test_real_model_starting_material_does_not_qualify PASSED
literature-classifier/tests/test_integration.py::test_real_model_negation_case PASSED
literature-classifier/tests/test_unit.py::test_sentence_splitting_abbreviations PASSED
literature-classifier/tests/test_unit.py::test_word_boundary_matching PASSED
literature-classifier/tests/test_unit.py::test_synonym_matching PASSED
literature-classifier/tests/test_unit.py::test_confidence_symmetry PASSED
literature-classifier/tests/test_unit.py::test_evidence_rule_table_all_rows PASSED
literature-classifier/tests/test_unit.py::test_no_abstract_insufficient_text PASSED
literature-classifier/tests/test_unit.py::test_missing_doi_pmid_handling PASSED
literature-classifier/tests/test_unit.py::test_criterion_normalisation PASSED
literature-classifier/tests/test_unit.py::test_chemical_punctuation_regex PASSED
literature-classifier/tests/test_unit.py::test_evidence_integrity_error_on_tampering PASSED
literature-classifier/tests/test_unit.py::test_criterion_conditional_contrast_guard PASSED
literature-classifier/tests/test_unit.py::test_insufficient_text_null_scores PASSED
============================= 24 passed in 25.33s =============================
```

---

## Empirical Evaluation & Statistical Rigor

### Holdout Evaluation on Entity B (Vanillin) with 95% Bootstrap CIs

Evaluation was performed on 20 unseen publications (8 Qualifiers, 12 Non-Qualifiers) using parameters tuned exclusively on Entity A ($`\tau = 0.65`$, $`\delta = 0.05`$):

| Metric | Point Estimate | 95% Bootstrap Confidence Interval ($`n=2000`$) |
| :--- | :---: | :---: |
| **Precision (Qualifies)** | **0.750** | **[0.400, 1.000]** |
| **Recall (Qualifies)** | **0.750** | **[0.400, 1.000]** |
| **F1 Score (Qualifies)** | **0.750** | **[0.400, 1.000]** |
| **Overall Accuracy** | **0.800** | — |
| **Evidence Correctness** | **85.0%** | — |

**Confusion Matrix:** $`\text{TP} = 6`$, $`\text{FP} = 2`$, $`\text{TN} = 10`$, $`\text{FN} = 2`$.

> [!NOTE]
> **Why Lead with Precision over Recall?**  
> In zero-shot cross-encoder NLI, un-tuned models achieve trivial 100% recall simply by predicting `qualifies` for any paper mentioning the target entity. In chemical and biological literature curation, false positives pollute downstream reaction databases. Reporting Precision with 95% confidence intervals demonstrates true discrimination power.

---

### Threshold Saturation & Margin Value Histogram

An empirical analysis of candidate sentence margins reveals significant threshold saturation:

```
MARGIN VALUE HISTOGRAM (Best Candidate Margin m in [-1, 1]):
Bin            | Count | Distribution
--------------------------------------------------
[-1.0, -0.6)   |     0 | 
[-0.6, -0.2)   |     0 | 
[-0.2, +0.2)   |     2 | ####
[+0.2, +0.6)   |     3 | ######
[+0.6, +1.0]   |    15 | ##############################
```

**Technical Finding:** 75% (15/20) of biomedical papers mentioning the entity fall into the $`[+0.6, +1.0]`$ margin bin because DeBERTa associates research discourse with positive claims. Setting $`\tau = 0.65`$ operates above the background saturation noise, successfully separating true producers from incidental mentions.

---

### Criterion Phrasing Sensitivity & Ensemble Impact

Evaluating natural-language phrasings across the holdout set demonstrates how the paraphrase ensemble stabilizes classification:

| Criterion Phrasing | Single Baseline Precision | Paraphrase Ensemble Precision | 95% Bootstrap CI | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phrasing 1 (Standard Passive):** `"Vanillin is synthesised, biosynthesised, or produced in this work"` | 0.400 | **0.545** (+14.5%) | [0.25, 0.85] | 0.750 | 0.632 |
| **Phrasing 2 (Agentive Active):** `"The authors synthesized, biosynthesized, or produced vanillin in this study."` | 0.700 | **0.750** (+5.0%) | [0.40, 1.00] | 0.750 | 0.750 |
| **Phrasing 3 (Product-Centric):** `"Vanillin was obtained as a synthesized or biosynthesized product."` | 0.444 | **0.667** (+22.3%) | [0.33, 1.00] | 0.750 | 0.706 |

---

### Tuning on Entity A (Psilocybin)

Grid search over $`\tau \in [0.10, 0.70]`$ and $`\delta \in [0.00, 0.30]`$ on 20 papers:
- Optimal tuned: $`\tau^* = 0.65`$, $`\delta^* = 0.05`$.
- Tuning metrics: Precision = 0.667, Recall = 1.000, F1 = 0.800.

---

## Case Studies & Categorized Error Analysis

### Case 1: Starting Material (Reagent Fallacy Solved)
- **Paper:** `"Compound 4 was prepared using psilocybin as a starting material."`
- **Behavior:** Criterion entailment is moderate ($`P_{\text{ent}} = 0.789`$), but affirmative contrast hypothesis (`"Psilocybin is a starting material or reactant in this reaction."`) scores $`P_{\text{ent}} = 0.977`$.
- **Outcome:** Contrast guard fires ($`0.977 \gt 0.789 + 0.05`$). Classified as `does_not_qualify` with `evidence_status: "contrast_mention"`.

### Case 2: Dietary Administration vs Synthesis
- **Paper:** PMID `42655822` (*Vanillin Improves Antioxidant Capacity in Broilers*)
- **Behavior:** Broilers were fed diets supplemented with vanillin. Affirmative contrast hypothesis (`"Vanillin was administered, consumed, or given to subjects in this study."`) scores $`P_{\text{ent}} = 0.945`$.
- **Outcome:** Contrast guard suppresses the false positive. Classified as `does_not_qualify`.

### Case 3: Negation Sensitivity
- **Paper:** `"No synthesis of psilocybin was observed."`
- **Behavior:** Negative clause triggers DeBERTa contradiction ($`P_{\text{con}} = 0.985`$), driving margin $`m = -0.970`$.
- **Outcome:** Classified as `does_not_qualify` with symmetric high confidence ($`0.98`$).

---

## User Feedback & Continuous Improvement Plan

1. **Active Learning Queue:** Predictions with `medium` or `low` confidence bands and runner-up margin gaps $`\lt 0.10`$ are flagged for human review.
2. **Persistence:** `POST /feedback` appends verified labels to `feedback.jsonl`.
3. **Logistic Meta-Calibration:** User feedback fits an empirical logistic regression over the feature vector below, providing well-calibrated probabilities $`P(\text{qualifies} \mid x)`$.

```math
x = \left[\, m_{\text{best}},\ c_{\text{best}},\ \mathbb{1}_{\text{title}},\ n_{\text{mentions}},\ \Delta m_{\text{runner-up}} \,\right]
```

---

## Assumptions, Limitations & Future Work

1. **Text Scope:** Currently processes titles, abstracts, and open-access full-text XML. Paywalled full text is inaccessible via Europe PMC REST API.
2. **Chemical Dependency Parsing:** Integrating ChemDataExtractor or spaCy-based chemical dependency parsing could explicitly verify syntactic head/dependent relationships for reactant vs product roles.
3. **Domain Fine-Tuning:** Fine-tuning DeBERTa-v3 on scientific bio-NLI datasets (such as BioNLI or ChemProt) would further tighten decision boundaries beyond zero-shot MNLI.
