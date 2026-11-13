# Mini Literature Classification Engine (`v3.2`)

**Candidate:** Ruhaan Kakar  
**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**Task:** Mini Literature Classification Engine (Technical Assessment)  
**System Scope:** Europe PMC automated literature mining, semantic Natural Language Inference (NLI) classification (`qualifies` / `does_not_qualify` / `insufficient_evidence`), multi-condition verification (Entity, Activity, System), G1/G2/G3 criterion grammar, structural title isolation, byte-exact verbatim evidence offsets, two-map MLE Platt confidence calibration, deterministic ranking, and production FastAPI service.  
**Execution Runtime:** Python 3.11, PyTorch 2.6.0, Hugging Face Transformers, FastAPI, NLTK, AnyIO, Uvicorn  

---

## Table of Contents
1. [Executive Summary & v3.2 Architecture Highlights](#1-executive-summary--v32-architecture-highlights)
2. [Guide for the Research Scientist & Bioinformatician](#2-guide-for-the-research-scientist--bioinformatician)
   - [2.1 The Scientific Problem This Solves](#21-the-scientific-problem-this-solves)
   - [2.2 Granular Breakdown of API Features](#22-granular-breakdown-of-api-features)
   - [2.3 Reason Code Registry & Three-Class Outcomes](#23-reason-code-registry--three-class-outcomes)
   - [2.4 Deliberate Deviations (D1–D6)](#24-deliberate-deviations-d1d6)
   - [2.5 Honest Shortcomings & Technical Caveats](#25-honest-shortcomings--technical-caveats)
3. [Step-by-Step Implementation & Run Guide](#3-step-by-step-implementation--run-guide)
   - [3.1 System Prerequisites](#31-system-prerequisites)
   - [3.2 Installation & Setup](#32-installation--setup)
   - [3.3 Starting the FastAPI Service](#33-starting-the-fastapi-service)
   - [3.4 Interacting with the API](#34-interacting-with-the-api)
   - [3.5 Running the Automated Test Suite](#35-running-the-automated-test-suite)
   - [3.6 Offline Batch Processing via CLI (`--pipeline v3` / `--pipeline v2`)](#36-offline-batch-processing-via-cli)
4. [Pipeline Architecture & 12-Rule Combiner](#4-pipeline-architecture--12-rule-combiner)
5. [Criterion Parser & Enumerated Grammar (G1 / G2 / G3)](#5-criterion-parser--enumerated-grammar-g1--g2--g3)
6. [Scoring, Calibration, and Ranking Specification](#6-scoring-calibration-and-ranking-specification)
7. [Evaluation Strategy & Curator Feedback Queue](#7-evaluation-strategy--curator-feedback-queue)
8. [Repository File & Directory Map](#8-repository-file--directory-map)

---

## 1. Executive Summary & v3.2 Architecture Highlights

The **Mini Literature Classification Engine** is a specialized, production-grade biomedical literature mining system built for **Loopworm Private Limited**. Its core mission is to automatically retrieve scientific publications from the public **Europe PMC** repository for a given biological or chemical entity and determine whether each publication **genuinely satisfies a specific natural-language criterion** (such as microbial biosynthesis or chemical synthesis) versus **merely mentioning the entity in passing**.

### Core Architecture Highlights (`v3.2`)

- **Criterion-Aware Multi-Condition Decomposition:** Decomposes user queries into three independent conditions:
  1. **C1 Entity:** Entity matching with lookaround regexes, length $\ge 5$, related compound guards, and prepositional role polarity (plus **R3b** synthesis-conflict routing).
  2. **C2 Activity:** Clause-local competing-hypothesis NLI over requested activity vs. alternatives (with IUPAC bracket-depth clause splitting, backward chemical anaphora resolution, and G3 numeric/exclusion verification).
  3. **C3 System:** Host organism verification against `config/systems.json` gazetteer with comparative-clause exclusion (`Unlike E. coli, yeast...`), `in_vitro` separation, and **D6** strict abstract contradiction handling.
- **Three-Class Outcomes ("Never Merge Unknown with No"):**
  - `qualifies`: The publication affirmatively satisfies all required conditions with verbatim evidence.
  - `does_not_qualify`: The publication is affirmatively refuted by a rule veto or an evidence-based negative (`wrong_system`, `different_activity`, `mention_only`, `background_citation`, `entity_absent`).
  - `insufficient_evidence`: The text does not contain enough information to decide (never converted into a false negative).
- **Enumerated Criterion Grammar (G1 / G2 / G3):**
  - **G1 (Product):** `<paper> reports <activity> of <entity> [in <system>] [in this study]`
  - **G2 (Agent):** `<paper> uses <entity> as a <catalyst|reagent|substrate> [in <activity>]`
  - **G3 (Modifiers):** Exclusions (*"not in yeast"*), unit-normalized numeric thresholds (*"> 5 g/L"*), and compound clauses.
  - Unsupported clauses trigger clean HTTP 422 errors by default; opt-in `allow_partial_criterion=true` evaluates the supported core and caps qualifications to `insufficient_evidence` (`unsupported_criterion_clause`).
- **Structural Title Isolation & Verbatim Slice Invariant:** Titles are never merged into scored body text; every evidence span asserts `canon_body[char_start:char_end] == evidence.display_text`.
- **Two-Map MLE Platt Confidence Calibration:** Fits regularized Newton-Raphson MLE Platt curves on dev LOEO folds (`config/calibration.json`) for `qualifies` and evidence-based negatives (`different_activity`, `wrong_system` only). Vetoes, `mention_only`, `background_citation`, and abstentions strictly emit `confidence: null` (**D5**).
- **Zero Paid LLM APIs:** 100% local inference with pinned Hugging Face revision (`fa280487`) and freely accessible public scientific APIs.

---

## 2. Guide for the Research Scientist & Bioinformatician

### 2.1 The Scientific Problem This Solves

In automated bio-curation pipelines (e.g. identifying microbial hosts that synthesize target metabolites, discovering novel enzyme pathways, or mining natural product titers), standard keyword search and generic embedding models fail due to **topical saturation without semantic satisfaction**:

1. **The Reagent Fallacy:** A paper describing the chemical synthesis of a novel dye using vanillin as a precursor mentions "vanillin" and "synthesis" dozens of times, tricking naive keyword search into classifying it as a vanillin production paper.
2. **The Epistemological Review Trap:** Systematic reviews stating *"we synthesize current knowledge on methylxanthine pharmacology"* trigger synthesis keywords without describing experimental production.
3. **Background Citation Leakage:** Introduction sections frequently cite prior art (*"Compound X has been produced by engineered yeast in prior studies"*) in papers that actually study downstream formulation or unrelated pathways.

---

### 2.2 Granular Breakdown of API Features

When querying `POST /classify`, the researcher receives a structured response conforming to the v3 specification:

| Feature | Field in Response | How It Works & Scientific Value |
|---|---|---|
| **Enumerated Grammar Spec** | `criterion_spec` | Structured parse representing entity role (`product` vs `agent`), agent kind, activity, target system, and scope. |
| **Dual Retrieval Tracking** | `retrieved_by` | Tracks whether the publication was found via `intent`, `plain` entity search, or `both`. |
| **Condition Decomposition** | `conditions: {entity, activity, system}` | Detailed status (`supported`, `contradicted`, `unknown`, `not_required`), raw feature score, also-mentioned organisms, and condition evidence spans. |
| **Machine-Readable Reason Code** | `reason_code` | Discloses the exact registered reason code for any non-qualifying or abstained result (e.g. `reagent_role`, `system_mismatch_abstract`). |
| **Exact Verbatim Evidence** | `evidence: {raw_text, display_text, raw_start, raw_end, role, verifiable_alone}` | Returns exact verbatim evidence with character slice offsets guaranteed byte-for-byte in canonical text. |
| **Source Record Audit Hash** | `source_record_sha256` | SHA-256 hash of the uncleaned raw record for verifiable reproducibility and auditing. |
| **Monotone Calibrated Confidence** | `confidence: {value, band, method, calibration_version}` | Honest uncertainty estimation; null on abstentions and uncalibrated negatives. |

---

### 2.3 Reason Code Registry & Three-Class Outcomes (18 Registered Codes)

Every non-qualifying or abstained result carries a registered reason code from `app/reasons.py` (17 reachable active codes plus deprecated `system_mismatch_abstract`):

| Reason Code | Label | Kind | Description |
|---|---|---|---|
| `no_text` | `insufficient_evidence` | Abstain | Document has no usable abstract or full text. |
| `abstract_only` | `insufficient_evidence` | Abstain | Abstract lacks decisive details; full-text is unavailable. |
| `mention_only` | `does_not_qualify` | Evidence-based | Entity is mentioned topically without any synthesis/production activity claim. |
| `activity_unclear` | `insufficient_evidence` | Abstain | Activity claims are ambiguous, hedged, or below NLI margin $\tau$. |
| `entity_unclear` | `insufficient_evidence` | Abstain | Target entity was not matched in the abstract (may be in Methods). |
| `entity_absent` | `does_not_qualify` | Evidence-based | Confirmed absent from complete full text. |
| `system_unstated` | `insufficient_evidence` | Abstain | Criterion requires host system, but abstract does not state one. |
| `system_mismatch_abstract` | `insufficient_evidence` | Abstain | *Deprecated under D6*: retained for compatibility; never emitted at runtime. |
| `conflicting_evidence` | `insufficient_evidence` | Abstain | Reagent role pattern detected in one clause alongside a synthesis claim elsewhere (**R3b**). |
| `evidence_split` | `insufficient_evidence` | Abstain | Activity sentence refers to entity via unresolved reference without joint binding. |
| `unsupported_criterion_clause` | `insufficient_evidence` | Abstain | Unparsed criterion clauses present under `allow_partial_criterion`. |
| `unstated_threshold` | `insufficient_evidence` | Abstain | Numeric G3 constraint requested (e.g. `> 5 g/L`) with no numeric measurement reported. |
| `different_activity` | `does_not_qualify` | Evidence-based | Verbatim this-work evidence proves a different activity or fails G3 threshold/exclusion (**D5** calibrated). |
| `wrong_system` | `does_not_qualify` | Evidence-based | Explicit conflicting host organism confirmed under **D6** (**D5** calibrated). |
| `reagent_role` | `does_not_qualify` | Veto | Entity is the object of an explicit prepositional role pattern with no conflicting synthesis claim (**R3a**). |
| `background_citation` | `does_not_qualify` | Evidence-based | Activity mention cites prior art below `P_THIS_WORK_MIN`. |
| `possible_review` | `insufficient_evidence` | Abstain | Review lexical score combined with low experimental attribution cues. |
| `review_article` | `does_not_qualify` | Veto | Europe PMC metadata `pubType=review` with `scope == "this_study"`. |

---

### 2.4 Deliberate Deviations (D1–D6)

| ID | Deviation | Rationale | Architectural Mitigation |
|---|---|---|---|
| **D1** | Qualifier Platt MLE includes borderline negatives within `NEG_FIT_WINDOW = 0.15` below $\tau$. | Fitting exclusively on far positives inflates Platt slope $A$ upward. | Prevents slope collapse (`slope_unstable_flag=false` in dev LOEO). |
| **D2** | Runtime assertion `canon_body[start:end] == evidence.display_text` with separate `canon_title`. | Eliminates paraphrased evidence and title leakage completely. | Titles tokenized separately and excluded from candidate selection. |
| **D3** | Externalized constants in `config/constants.md` with `splits_sha256` verification. | Prevents hidden heuristics and dev/test contamination. | Verified via automated CI doc-drift checks (`tests/unit/test_doc_drift.py`). |
| **D4** | Strict two-veto ceiling (`reagent_role`, `review_article`). | Prevents ad-hoc regex veto sprawl. | Co-occurring reagent + synthesis clauses route to `conflicting_evidence` (**R3b**). |
| **D5** | Calibrated negative Platt map fitted only on `different_activity` and `wrong_system`; all other non-qualifiers return `confidence: null`. | Vetoes, background mentions, and abstentions are discrete rule outcomes, not calibrated margin posteriors. | Schema validator rejects non-null confidence on `insufficient_evidence`, vetoes, `mention_only`, and `background_citation`. |
| **D6** | Explicit host contradictions in abstracts route to `does_not_qualify(wrong_system)`. | Avoids abstention inflation when an abstract explicitly states an incompatible host (e.g. *Artemisia annua* vs yeast). | Monitored via stratified `clean` vs `noisy` abstention rates. |

---

### 2.5 Honest Shortcomings & Technical Caveats

1. **Abstract Paywall Truncation:** In scientific publishing, fermentation yields and synthetic methods often appear only in paywalled full text. Because the engine strictly refuses to invent evidence, papers with vague abstracts (*"We investigated yeast metabolism"*) will abstain as `insufficient_evidence`.
2. **Hyphenated Stereoisomers:** Word-boundary lookaround regexes treat hyphens as boundary characters. Locant-prefixed isomers (e.g. *o-vanillin*) may trigger initial candidate matching, leaving fine disambiguation to downstream NLI. Integrating chemical Named Entity Recognition (NER) is recommended for stereochemical isolation.
3. **Multi-Sentence Synthetic Narratives:** When an organic total synthesis is described across multiple paragraphs without an overarching summary sentence, the engine evaluates single sentences and selects the single highest-scoring sentence rather than aggregating cross-paragraph reasoning.

---

## 3. Step-by-Step Implementation & Run Guide

### 3.1 System Prerequisites

- **OS:** Windows 10/11, Ubuntu 20.04+, macOS
- **Python:** Python 3.11 (tested on 3.11.0)
- **RAM:** 4 GB minimum (8 GB+ recommended)
- **Accelerator:** CPU-only supported; CUDA GPU accelerates batch NLI.

---

### 3.2 Installation & Setup

```bash
cd loopworm/literature-classifier

# Create virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Bootstrap NLTK Punkt tokenizer resources
python -m app.setup_nltk
```

---

### 3.3 Starting the FastAPI Service

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

### 3.4 Interacting with the API

#### Interactive Swagger Documentation:
Open your browser and navigate to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "entity": "artemisinic acid",
       "criterion": "The paper reports microbial biosynthesis of artemisinic acid in yeast in this study.",
       "max_results": 10,
       "expand_aliases": true,
       "pipeline": "v3"
     }'
```

#### Curator Feedback Submission:
```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "paper_id": "42107802",
       "entity": "artemisinic acid",
       "criterion": "The paper reports microbial biosynthesis of artemisinic acid in yeast in this study.",
       "suggested_label": "qualifies",
       "notes": "Verified 25 g/L yeast titer paper."
     }'
```

---

### 3.5 Running the Automated Test Suite

```bash
# Tier 1: Fast deterministic unit tests (including v3 remediation suite)
python -m pytest tests/unit -v

# Tier 2: Real DeBERTa-v3 model fixture tests (Q1-Q3, N1-N6, D1)
python -m pytest tests/model -v

# Tier 3: Live Europe PMC network integration test
python -m pytest tests/live -m live -v

# Run entire test suite:
python -m pytest tests/unit tests/model tests/live -v
```

---

### 3.6 Offline Batch Processing via CLI

The CLI supports both `--pipeline v3` (default multi-condition 3-class) and `--pipeline v2` (legacy sentence-level binary):

```bash
# Run v3 pipeline offline on cached snapshots:
python -m app.cli --entity vanillin \
                  --criterion "The paper describes the synthesis or production of vanillin." \
                  --pipeline v3 \
                  --offline \
                  --output sample_output.json

# Run legacy v2 pipeline:
python -m app.cli --entity vanillin \
                  --criterion "The paper describes the synthesis or production of vanillin." \
                  --pipeline v2 \
                  --offline \
                  --output v2_output.json
```

---

## 4. Pipeline Architecture & 12-Rule Combiner

The v3.2 decision layer executes a strict **12-rule** precedence combiner (`RULES` in `app/combiner.py`: `R1`–`R11` with `R3a`/`R3b`; first matching rule wins):

```text
R1  no_text                       -> insufficient_evidence (no_text)
R2  review_veto                   -> does_not_qualify (review_article)
R3a reagent_veto (no conflict)    -> does_not_qualify (reagent_role)
R3b conflicting_evidence          -> insufficient_evidence (conflicting_evidence)
R4  entity_absent / unclear       -> does_not_qualify (entity_absent) | insufficient_evidence (entity_unclear)
R5  background_citation / mention -> does_not_qualify (background_citation | mention_only)
R6  wrong_system                  -> does_not_qualify (wrong_system)
R7  system_unstated               -> insufficient_evidence (system_unstated)
R8  different_activity / G3 fail  -> does_not_qualify (different_activity)
R9  unstated_threshold / unclear  -> insufficient_evidence (unstated_threshold | activity_unclear)
R10 unsupported_criterion_clause  -> insufficient_evidence (unsupported_criterion_clause)
R11 all_satisfied                 -> qualifies (None)
```

---

## 5. Criterion Parser & Enumerated Grammar (G1 / G2 / G3)

The engine enforces a published grammar. Unrecognized clauses raise HTTP 422 errors unless `--allow-partial-criterion` is supplied:

- **G1 (Product Role):**  
  `"<paper> reports <activity> of <entity> [in|using|by <system>] [in this study]"`  
  *Examples:*
  - *"The paper describes the synthesis or production of vanillin."*
  - *"The paper reports microbial biosynthesis of artemisinic acid in yeast in this study."*
  - *"The paper reports enzymatic production of vanillin in vitro."*
- **G2 (Agent Role):**  
  `"<paper> uses <entity> as a <catalyst|reagent|substrate> [in <activity>]"`  
  *Example:* *"The paper uses palladium(II) acetate as a catalyst in cross-coupling reactions."*
- **G3 (Modifiers):**  
  Supports host exclusions (*"not in yeast"*), unit-normalized numeric thresholds (*"> 5 g/L"*, *"yield >= 80%"*), and compound clauses. Unrecognized non-grammar clauses (e.g. *"in human clinical subjects"*) return HTTP 422 by default.

---

## 6. Scoring, Calibration, and Ranking Specification

### Per-Condition Scoring
- **C1 Entity Score:** `1.0` if matched, `-0.95` if reagent role triggered, `0.0` if abstract-only unknown.
- **C2 Activity Score:** Raw NLI margin $P(\text{entail}) - P(\text{contradict})$ against competing hypotheses and clause-local $p_{\text{this\_work}}$.
- **C3 System Score:** `1.0` if host matched in gazetteer (after comparative-clause exclusion), `-0.90` if contradicted (**D6**), `0.0` if unstated.

### Two-Map MLE Platt Scaling Calibration (`config/calibration.json`)
`app/scoring.py` loads fitted regularized logistic MLE parameters produced by `eval/calibrate.py` across the fixed 4-entity dev split (`eval/splits.json`):
$$P(\text{correct}) = \frac{1}{1 + \exp(-(A \cdot x + B))}$$
- **Qualifier path (`qualifies`):** uses `(A, B)` fitted on dev positives + borderline negatives within `NEG_FIT_WINDOW = 0.15`.
- **Negative path (`different_activity`, `wrong_system` only):** uses `(A_neg, B_neg)` fitted on dev evidence-based negative margins (**D5**).
- **Confidence bands:** High ($P \ge 0.80$), Medium ($0.50 \le P < 0.80$), Low ($P < 0.50$). Vetoes, `mention_only`, `background_citation`, and abstentions emit `confidence: null`.

---

## 7. Evaluation Strategy & Curator Feedback Queue

1. **Leave-One-Entity-Out (LOEO) & Strict Held-Out Test Split (`eval/splits.json`):** Dev entities (`vanillin`, `caffeine`, `curcumin`, `chitin`) are strictly disjoint from held-out test entities (`palladium(II) acetate`, `artemisinic acid`, `lipase`).
2. **Stratified `clean` + Real Europe PMC `noisy` Benchmark (`152` records):** Reproducibly audited via SHA-256 `data/snapshots/manifest.json` and executed with real `DebertaNLIScorer` via `python -m eval.run --pipeline both --split test` and [`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb).
3. **Technical Report:** Full challenge–response summary, v2 vs. v3.2 test metrics, calibration audit, and case studies live in [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).
4. **Curator Feedback Queue:** `POST /feedback` atomically appends corrections to `feedback.jsonl`. Run `python -m eval.ingest_feedback` to stage reviewed entries into `eval/candidate_feedback.json`.

---

## 8. Repository File & Directory Map

```text
loopworm/
├── literature-classifier/
│   ├── app/
│   │   ├── main.py            # FastAPI service (/classify, /feedback, /health), concurrency semaphore
│   │   ├── schemas.py         # Pydantic v2 models enforcing D5 confidence & reason registry
│   │   ├── classifier.py      # Orchestrator (--pipeline v3 / v2), structural title isolation, slice invariant
│   │   ├── criterion_parser.py# Scoped criterion parser for G1/G2/G3 grammars & Systems gazetteer
│   │   ├── conditions.py      # C1 Entity, C2 Activity, C3 System condition checkers
│   │   ├── combiner.py        # Strict 12-rule deterministic combiner (R1-R11 + R3a/R3b)
│   │   ├── reasons.py         # Reason code registry (18 registered codes, D6 deprecation)
│   │   ├── hypotheses.py      # Intent detection, dynamic hypothesis builder, role pattern matcher
│   │   ├── nli.py             # Local Cross-Encoder DeBERTa-v3 wrapper & in-memory StubScorer
│   │   ├── scoring.py         # Two-map MLE Platt loader (config/calibration.json), bands, ranking
│   │   ├── retrieval.py       # Europe PMC REST client, dual query retrieval, deduplication
│   │   ├── text_utils.py      # Canonicalization, bracket-depth clause split, anaphora, alias vetting
│   │   ├── setup_nltk.py      # NLTK resource bootstrap (punkt & punkt_tab)
│   │   └── cli.py             # Reproducible offline/online CLI tool (--pipeline v3 / v2)
│   ├── config/
│   │   ├── systems.json       # Systems gazetteer (yeast, bacteria, fungi, plant, mammalian, in_vitro)
│   │   ├── constants.md       # Hand-set constants register & LOEO sensitivity analysis
│   │   ├── calibration.json   # Fitted two-map MLE Platt parameters (splits_sha256 verified)
│   │   ├── intent_map.json    # Intent classification triggers, search terms, production verbs
│   │   ├── role_patterns.json # Prepositional role regex patterns (modifier-tolerant)
│   │   ├── templates.json     # True-paraphrase synthesis hypotheses and negative templates
│   │   └── alias_denylist.txt # Denylisted alias collisions and ambiguous short tokens
│   ├── data/snapshots/        # SHA-256 audited offline Europe PMC & PubChem snapshots (manifest.json)
│   ├── eval/
│   │   ├── splits.json        # Authoritative dev/test entity splits
│   │   ├── calibrate.py       # Regularized Newton-Raphson MLE Platt fitting script
│   │   ├── run.py             # Offline stratified v2 vs. v3.2 evaluation runner
│   │   ├── ingest_feedback.py # Curator feedback queue validation script
│   │   ├── metrics.py         # 3-class metrics, Wilson CIs, McNemar test, ECE, Cohen's kappa
│   │   └── labels.csv         # 152-row clean + noisy ground truth dataset
│   ├── notebooks/
│   │   └── evaluation.ipynb   # Executed offline reproducible DeBERTa evaluation notebook
│   ├── tests/                 # Tier 1 unit (50 tests), Tier 2 real DeBERTa model (12 tests), Tier 3 live
│   ├── TECHNICAL_REPORT.md    # Brief technical report (improvements, metrics, limitations)
│   ├── sample_output.json     # Canonical sample JSON payload (v3.2 schema)
│   └── README.md              # Technical documentation and user guide
├── ARCHITECTURE.md            # Detailed system design specification
├── KNOWN_ERRORS.md            # Empirical error taxonomy and failure mode case studies
├── TECHNICAL_REPORT.md        # Mirrored technical report
├── implementation.md          # Canonical v3.2 implementation plan specification
└── Project.md                 # Master project documentation
```

---

*Compiled, tested, and verified against the live environment.*
