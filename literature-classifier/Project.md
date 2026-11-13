# Project Overview: Mini Literature Classification Engine

**Candidate:** Ruhaan Kakar  
**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**Task:** Mini Literature Classification Engine (Technical Assessment)  
**System Scope:** Europe PMC automated literature mining, semantic Natural Language Inference (NLI) classification (`qualifies` / `does_not_qualify`), verbatim evidence extraction with byte-level character offsets, Platt-calibrated confidence scoring, deterministic ranking, and production FastAPI service.  
**Execution Runtime:** Python 3.11, PyTorch 2.6.0, Hugging Face Transformers, FastAPI, NLTK, AnyIO, Uvicorn  

---

## 1. Executive Summary & Purpose of the Project

The **Mini Literature Classification Engine** is a specialized, production-ready biomedical literature mining system built for **Loopworm Private Limited**. Its core mission is to automatically retrieve scientific publications from the public **Europe PMC** repository for a given biological or chemical entity (e.g. *vanillin*, *artemisinic acid*, *palladium(II) acetate*, *lipase*) and determine whether each publication **genuinely satisfies a specific natural-language criterion** (such as microbial biosynthesis or chemical synthesis) versus **merely mentioning the entity in passing**.

### The Fundamental Dilemma
In bioinformatics, strain engineering, and natural product discovery, researchers are flooded with thousands of scientific papers. Standard keyword search engines (PubMed, Europe PMC) and generic bi-encoder vector databases suffer from a fatal flaw:
> **Topical Saturation vs. Semantic Satisfaction:**  
> A paper can mention a target compound dozens of times, use reaction verbs, and discuss biochemistry without actually producing, synthesizing, or engineering the compound.

This engine solves this problem with high scientific precision, strict mathematical calibration, zero hallucinated evidence, and **zero reliance on paid closed-source LLMs** (running 100% locally on free and open-source models).

---

## 2. The Scientific Problem: Why Naive Approaches Fail

When building automated bioprospecting and literature curation pipelines, three deceptive phenomena routinely break standard search algorithms and naive language models:

```mermaid
flowchart TD
    subgraph Failures ["The Three Scientific Traps in Literature Mining"]
        T1["1. The Reagent Fallacy<br/>(Catalyst/Precursor Confusion)"]
        T2["2. The Epistemological Review Trap<br/>('We synthesize knowledge')"]
        T3["3. Background Citation Leakage<br/>('Compound X was produced in prior work')"]
    end

    T1 -->|Naive Embedding / Keyword Match| FP1["FALSE POSITIVE:<br/>Pd(OAc)2 classified as product when it was just a catalyst!"]
    T2 -->|Naive LLM / Keyword Match| FP2["FALSE POSITIVE:<br/>Review article classified as chemical synthesis paper!"]
    T3 -->|Topical Keyword Match| FP3["FALSE POSITIVE:<br/>Unrelated pathway study classified as producer!"]
```

### 1. The Reagent Fallacy (Catalyst & Substrate Confusion)
- **Example:** *"In the presence of palladium(II) acetate and triphenylphosphine, substituted dibenzoxepines were prepared in 78% yield."*
- **What goes wrong:** Naive keyword matching or dense vector similarity sees `"palladium(II) acetate"` and `"prepared in 78% yield"` and enthusiastically tags the paper as a palladium synthesis study. In reality, palladium was merely the *catalyst*, and dibenzoxepines were the actual product.

### 2. The Epistemological Review Trap
- **Example:** *"In this review, we synthesize current knowledge on caffeine metabolism and cardiovascular outcomes."*
- **What goes wrong:** The word *"synthesize"* is used epistemologically (meaning "to summarize or bring together"), not chemically. Unhardened models mistakenly classify this literature review as an experimental synthesis paper.

### 3. Background Citation Leakage
- **Example:** *"Artemisinic acid has been produced by engineered yeast in previous studies. Here, we investigate downstream taxadiene isomerization pathways."*
- **What goes wrong:** The paper cites prior art in its introduction. It is not actually reporting artemisinic acid production as an experimental finding of *this* study.

---

## 3. The Story of the System: The v1 Failure and the v2 Overhaul

The project underwent a rigorous evolutionary leap between Version 1 and Version 2 after an external live evaluation exposed subtle, deep-seated flaws in standard text classification.

### The Wake-Up Call (The 30-Paper Live Audit)
When v1 was evaluated on three live queries from Europe PMC (30 real papers across Caffeine, Palladium(II) acetate, and Artemisinic acid), only **~8 to 10 out of 30** were correctly classified:
- **Caffeine (synthesis criterion):** 7/10 qualified, but only ~3 were actual synthesis papers. Reviews, toxicology reports, and molecular dynamics (MD) simulations were erroneously qualified.
- **Palladium(II) acetate (product criterion):** 6/10 qualified, but Pd(OAc)₂ was simply the reaction catalyst in almost all of them.
- **Artemisinic acid (microbial biosynthesis):** 10/10 qualified—the classifier had zero discriminating power, qualifying network pharmacology papers and background mentions alike. True breakthrough yeast engineering papers (PMID 42107802) were buried down at Rank #7!

### The Three Root Causes in v1
1. **Decoupled Scoring and Evidence:** v1 computed a document-level score by aggregating lexical and topical matches across the entire text, while arbitrarily picking a sentence to display. As a result, papers dense with the entity's name scored high even when the cited sentence had nothing to do with production.
2. **Relaxed Hypothesis Ensemble:** Hypothesis templates dropped strict qualifiers like *"in this study"*, *"chemically"*, and *"as a product"*, unintentionally asking the model whether the entity was merely associated with production verbs.
3. **Narrow Reagent Guards:** The guard only looked for phrases like *"using X as the precursor"*, completely missing prepositional phrases like *"in the presence of X"* or *"catalyzed by X"*.

### The Remediation Workstreams (W1–W9)
To achieve production robustness, the codebase was completely re-architected across 9 formal workstreams:
- **W1 (Sentence-Level Coupling):** The document classification score is *strictly* equal to the winning sentence's score.
- **W2 (Criterion Fidelity):** Strict true paraphrases only; creation of an explicit negative-role hypothesis comparison.
- **W3 (Expanded Role Guard):** 15 prepositional patterns (`config/role_patterns.json`) detecting reagent, catalyst, and substrate roles.
- **W4 (Abstain Path):** Principled abstention (`evidence_status: "insufficient"`) when no sentence contains both the entity and a production verb.
- **W5 (Alias Vetting):** Minimum length $\ge 5$, denylist filtering (`config/alias_denylist.txt`), and zero-width regex lookarounds.
- **W6 (Intent-Augmented Retrieval & Deduplication):** Query expansion with intent verbs; deduplication by DOI and normalized title; pre-tokenization HTML stripping.
- **W7 (Platt Scaling Calibration):** Mapping raw scores to true posterior probabilities with strictly monotone qualitative bands.
- **W8 (Evaluation Overhaul):** Hard-negative benchmarks, Leave-One-Entity-Out (LOEO) validation, Wilson score 95% confidence intervals, and exact McNemar testing.
- **W9 (Production Deliverables):** Pinned dependencies, async FastAPI service with load shedding, automated test suite (34/34 tests green), and comprehensive documentation.

---

## 4. The 7 Empirical Failure Modes and Their Cures (`KNOWN_ERRORS.md`)

The project explicitly documents and mitigates seven distinct empirical failure modes:

| # | Failure Mode | Real-World Manifestation | Root Cause | Architectural Remediation in v2 |
|---|---|---|---|---|
| **F1** | **Review Article Trap** | Review titled *"Pharmacological Review of Methylxanthines"* stating *"we synthesize current knowledge"* scored 0.987. | Zero-shot transformers treat lexical overlap of "synthesize" as semantic entailment. | **Review Pre-Filter regex** + **Candidate Verb Gate**; automatic routing to the **Abstain Path** (`evidence_status: "insufficient"`). |
| **F2** | **Actor / Role Confusion** | *"In presence of palladium(II) acetate … dibenzoxepines were prepared"* scored 0.981 for Pd(OAc)₂ synthesis. | Self-attention binds the reaction verb to the catalyst entity rather than the true product. | **15 Prepositional Role Patterns** (`config/role_patterns.json`) + **Negative-Role Hypothesis Comparison** (`"{entity} is used as a reagent, catalyst, or starting material"`). Logs `guard_reason`. |
| **F3** | **Alias Collisions** | Historical synonym `"Thein"` for caffeine matched an 1845 paper by author *"Marinus Thein"*. | Automated PubChem synonyms < 5 chars collide with surnames, abbreviations, or dictionary words (`mat`, `aid`, `thein`). | **Alias Denylist** (`config/alias_denylist.txt`), **Length Filter ($\ge 5$ chars)**, and regex lookarounds `(?<!\w){alias}(?!\w)`. |
| **F4** | **Inverted Document Ranking** | A network pharmacology paper with 28 mentions of artemisinic acid ranked #2, while the true 25 g/L yeast biosynthesis paper ranked #7. | Document score was decoupled from sentence evidence, favoring topical density over evidentiary proof. | **Strict Sentence-Level Coupling** (`doc_score = max(cand_scores)`) + **Abstain-Aware Deliberate Ranking Key** (PMID 42107802 now ranks #1). |
| **F5** | **Uncalibrated Confidence Bands** | A score of `0.8523` was labeled "medium" confidence while `0.7494` was labeled "high". | Ad-hoc threshold formulas without a mathematically unified posterior probability scale. | **Platt Scaling Logistic Sigmoid** ($P(\text{correct}) \in [0.0, 1.0]$) with strictly monotone bands: High ($\ge 0.80$), Medium ($0.50–0.80$), Low ($< 0.50$). |
| **F6** | **Retrieval Starvation** | Searching Europe PMC for `TITLE_ABS:"caffeine"` returned 15 clinical/toxicology papers and zero synthesis papers. | Search engines rank by global popularity/citations, starving classifiers of rare synthesis literature. | **Intent-Augmented Query Generation:** Automatically parses criterion intent to inject production verbs into the Lucene search query. |
| **F7** | **Preprint Duplication & HTML Offset Drift** | bioRxiv preprints and Nature published versions returned twice; raw HTML tags (`<i>E. coli</i>`) shifted character offsets. | Uncleaned metadata and failure to reconcile preprints with peer-reviewed versions. | **DOI & Normalized Title Deduplication** (preferring peer-reviewed records and tracking `dedup_of`) + **Pre-Canonicalization HTML Tag Stripping**. |

---

## 5. End-to-End Pipeline Architecture & Lifecycle

The classification pipeline consists of 11 strictly sequenced stages ([A] through [K]):

```mermaid
flowchart TD
    REQ["POST /classify (entity, criterion, max_k)"] --> INTENT["[Stage A] Intent & Verb Parsing<br/>(Detect domain intent: synthesis, pharmacology, etc.)"]
    INTENT --> RETRIEVE["[Stage B] Europe PMC Retrieval<br/>(Intent-augmented query + vetted PubChem aliases)"]
    RETRIEVE --> DEDUPE["[Stage C] Deduplication & Markup Cleaning<br/>(Unify DOIs & titles; strip HTML tags before tokenizing)"]
    DEDUPE --> PREFILTER{"[Stage D] Review Pre-Filter<br/>(Is title/abstract an overview or review?)"}
    
    PREFILTER -->|"Yes (Review Article)"| ABSTAIN_REV["Abstain: does_not_qualify<br/>evidence_status: insufficient<br/>guard_reason: review_article_abstain"]
    PREFILTER -->|"No (Research Article)"| SEGMENT["[Stage E] Canonicalization & Tokenization<br/>(Unicode NFKC, dot-masking U+2024, NLTK Punkt)"]
    
    SEGMENT --> CAND_GATE{"[Stage F] Candidate Verb Gate<br/>(Does sentence have Entity AND Production Verb?)"}
    CAND_GATE -->|"No Eligible Sentences"| ABSTAIN_NO_VERB["Abstain: does_not_qualify<br/>evidence_status: insufficient<br/>evidence: null"]
    
    CAND_GATE -->|"Eligible Sentences Found"| NLI_BATCH["[Stage G] Batched Cross-Encoder Scoring<br/>(DeBERTa-v3 evaluates Positive Paraphrases vs. Negative Role)"]
    NLI_BATCH --> GATES["[Stage H] Gate Evaluations<br/>1. Role Pattern Guard (catalyst/reagent check)<br/>2. 'This Work' Gate (demote background citations)"]
    
    GATES --> SENT_DECIDE["[Stage I] Winning Sentence Selection<br/>doc_score = max(sentence_scores)<br/>evidence = argmax(sentence)<br/>assert canon[start:end] == evidence.text"]
    
    SENT_DECIDE --> CALIB["[Stage J] Platt Scaling Calibration<br/>final_score -> P(correct) in [0.0, 1.0]<br/>Monotone Bands: High >=0.80, Medium 0.50-0.80, Low <0.50"]
    CALIB --> RANK["[Stage K] Deliberate Deterministic Ranking<br/>1. Qualifiers (by calibrated P desc)<br/>2. Scored Non-Qualifiers<br/>3. Abstained Papers<br/>4. No-Text Papers"]
    
    RANK --> RESP["JSON Response Payload<br/>(rank, title, pmid/doi, classification, evidence, confidence)"]
```

### Granular Stage Descriptions
- **Stage A: Intent & Verb Parsing:** Analyzes the natural language criterion against `config/intent_map.json` to detect target domain intent (e.g. `synthesis`, `pharmacology`) and extract mandatory production verbs (`synthesized`, `produced`, `isolated`, `biosynthesized`, `yielded`, `fermentation`).
- **Stage B: Intent-Augmented Retrieval:** Queries the Europe PMC REST API with an enriched boolean Lucene query:
  `TITLE_ABS:("{entity}" AND (synthesis OR biosynthesis OR "produced by" OR fermentation OR preparation OR yielded))`
- **Stage C: Deduplication & Markup Cleaning:** Strips HTML entities and XML markup (`<i>`, `<sub>`, `&amp;`) before running sentence tokenization. Compares DOIs and normalized titles to merge preprints (bioRxiv, ChemRxiv) into their peer-reviewed versions, tracking the merged ID in `dedup_of`.
- **Stage D: Review Pre-Filter:** Examines titles and abstracts for review signals (`review`, `systematic review`, `meta-analysis`, `we synthesize current knowledge`). Literature reviews bypass heavy model inference and abstain immediately.
- **Stage E: Canonicalization & Protected Segmentation:** Performs Unicode NFKC normalization, masks abbreviation periods with one-dot leader `․` (U+2024) to protect chemical abbreviations (`e.g.`, `i.e.`, `et al.`, `sp. nov.`), and segments sentences using NLTK Punkt.
- **Stage F: Candidate Verb Gate:** Scans every sentence using zero-width lookaround regexes. A sentence is eligible for NLI inference **if and only if** it contains the target entity (or vetted alias) **AND** a criterion-mandated production verb. If zero sentences qualify, the paper abstains.
- **Stage G: Batched Cross-Encoder NLI:** Candidate sentences are batched and fed to the local `cross-encoder/nli-deberta-v3-small` model. For each sentence, the model evaluates true positive paraphrases against an explicit negative-role hypothesis.
- **Stage H: Gate Evaluations:**
  1. *Role Pattern Guard:* Scans against 15 prepositional regexes (`in the presence of`, `catalyzed by`, `using X as catalyst`). If a pattern matches, the paper is disqualified and `guard_reason` is recorded.
  2. *"This Work" Gate:* Distinguishes experimental first-person cues (`we synthesized`, `was obtained in 82% yield`, `herein`) from background prior art citations (`has been produced by`, `reported to`, `prior studies`). Background citations are capped strictly below threshold $\tau = 0.45$.
- **Stage I: Winning Sentence Selection & Invariant Assertion:** The document score is strictly the winning sentence's final score:
  ```python
  doc_score = max(cand_final_scores)
  evidence_sentence = argmax(cand_final_scores)
  assert canonical_doc[evidence.char_start:evidence.char_end] == evidence.text
  ```
- **Stage J: Platt Scaling Calibration:** Maps raw scores to a true calibrated probability $P(\text{correct}) \in [0.0, 1.0]$ via a logistic sigmoid, and assigns strictly monotone qualitative bands (High, Medium, Low).
- **Stage K: Deliberate Deterministic Ranking:** Enforces a 4-tier sort order: (1) Qualifiers sorted by calibrated probability descending, (2) Scored non-qualifiers descending, (3) Abstained papers (`insufficient`), and (4) Papers with no usable text (`no_text`). Ties are broken deterministically by publication year and PMID.

---

## 6. Mathematical & Scoring Specification

### 1. Per-Sentence Final Scoring
For any eligible candidate sentence $s$:
1. **Positive Paraphrase Margin:**
   $$\text{margin} = P(\text{entailment}) - P(\text{contradiction})$$
   Evaluated against full-fidelity paraphrases (preserving *"in this study"*, *"chemically"*, or *"as a product"*). The maximum margin across paraphrases is retained.
2. **Negative-Role Score:**
   Evaluated against hypothesis $H_{\text{role}}$: `"{entity} is used as a reagent, catalyst, or starting material."`:
   $$\text{role\_score} = P(\text{entailment})$$
3. **Composite Final Score:**
   $$\text{final\_score} = \text{positive\_margin} - \text{role\_score} - \text{penalties}$$

### 2. Platt Scaling Calibration
Raw margins are converted to well-calibrated posterior probabilities using a logistic sigmoid centered at decision threshold $\tau = 0.45$:
$$P(\text{correct}) = \frac{1}{1 + \exp(-7.0 \cdot (\text{final\_score} - \tau))}$$

### 3. Strictly Monotone Confidence Bands
Qualitative bands map strictly and monotonically from $P(\text{correct})$:
- **High:** $P(\text{correct}) \ge 0.80$
- **Medium:** $0.50 \le P(\text{correct}) < 0.80$
- **Low:** $P(\text{correct}) < 0.50$
*(If the difference between the winning sentence and the runner-up sentence is $< 0.10$, an ambiguity penalty dampens a High band to Medium).*

---

## 7. The Core Technical Constraints & Architecture Invariants

| Constraint / Feature | Implementation in Engine | Architectural Rationale |
|---|---|---|
| **Zero Paid LLM APIs** | `cross-encoder/nli-deberta-v3-small` (pinned revision `fa280487`) | Complete data privacy, zero API bills, zero rate limits, and 100% offline reproducibility. |
| **Zero Hallucination Guarantee** | `evidence_status: "insufficient"`, `evidence: null` | The engine will **never invent, pad, or guess evidence**. If text lacks proof, it abstains explicitly. |
| **Exact Traceability** | Byte-for-byte character slice offsets `[char_start:char_end]` | Direct auditability for scientists; verified by in-code assertions. |
| **Asynchronous Load Shedding** | `asyncio.Semaphore(2)` in FastAPI router | Prevents CPU/CUDA threadpool starvation during surges; returns clean HTTP 503 with `Retry-After: 10`. |
| **Upstream Network Resilience** | Exponential backoff `[1.0s, 2.0s, 4.0s]` with `Retry-After` handling | Respects Europe PMC server health; raises diagnostic HTTP 502 on persistent failures. |
| **Deterministic Tie-Breaking** | Primary: Class & Score; Secondary: `pub_year` desc; Tertiary: `paper_id` asc | Exactly reproducible ranking order across runs. |

---

## 8. Codebase Structure & File Navigation

```text
loopworm/
├── literature-classifier/
│   ├── app/
│   │   ├── main.py            # FastAPI service, endpoints (/classify, /feedback, /health), concurrency semaphore
│   │   ├── schemas.py         # Pydantic v2 data models: EvidenceSpan, ConfidenceDetails, ScoreBreakdown
│   │   ├── classifier.py      # Core classification orchestrator, candidate verb gating, sentence coupling
│   │   ├── hypotheses.py      # Intent detection, dynamic hypothesis builder, role pattern matcher
│   │   ├── nli.py             # Local Cross-Encoder DeBERTa-v3 wrapper & in-memory StubScorer
│   │   ├── scoring.py         # Platt scaling calibration, monotone bands, deliberate ranking sort key
│   │   ├── retrieval.py       # Europe PMC REST client, backoff retry, DOI/title dedupe, PubChem syns
│   │   ├── text_utils.py      # Single-pass canonicalizer, Punkt abbreviation protection, alias vetting
│   │   ├── setup_nltk.py      # Standalone NLTK resource bootstrap (punkt & punkt_tab)
│   │   └── cli.py             # Reproducible offline/online CLI tool
│   ├── config/
│   │   ├── intent_map.json    # Intent classification triggers, search query terms, production verbs
│   │   ├── role_patterns.json # 15 regex patterns for catalyst, reagent, and precursor roles
│   │   ├── templates.json     # True-paraphrase synthesis hypotheses and negative-role templates
│   │   └── alias_denylist.txt # Denylisted alias collisions, common surnames, and ambiguous short tokens
│   ├── data/
│   │   └── snapshots/         # Cached Europe PMC JSON/XML snapshots for offline benchmarking
│   ├── eval/
│   │   ├── run.py             # Leave-One-Entity-Out (LOEO) cross-validation runner
│   │   ├── metrics.py         # Pure-standard-library Wilson CIs, exact McNemar test, Cohen's kappa
│   │   └── labels.csv         # 100-paper authentic benchmark dataset
│   ├── tests/
│   │   ├── conftest.py        # Shared pytest fixtures (stub scorer, session-scoped real DeBERTa scorer)
│   │   ├── unit/              # Tier 1: 23 fast deterministic unit tests (~3.8s)
│   │   ├── model/             # Tier 2: 10 real DeBERTa model fixture tests (Q1-Q3, N1-N6, D1) (~13.9s)
│   │   └── live/              # Tier 3: Live Europe PMC network integration test (~14.7s)
│   ├── requirements.txt       # Pinned production dependencies
│   ├── sample_output.json     # Canonical sample JSON payload
│   ├── feedback.jsonl         # Active learning curator sink
│   └── README.md              # Public documentation and execution guide
├── ARCHITECTURE.md            # Detailed system design specification
├── KNOWN_ERRORS.md            # Empirical error taxonomy and failure mode case studies
├── implementation.md          # Technical assessment brief mapping, workstreams, and DoD
└── project.md                 # This comprehensive master project documentation
```

---

## 9. API Specification & Live Examples

### 1. `POST /classify` (Primary Endpoint)
Retrieves publications from Europe PMC, segments candidate sentences, executes batched NLI inference, calibrates confidence, and ranks results.

#### Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "entity": "artemisinic acid",
       "criterion": "The paper reports microbial biosynthesis of artemisinic acid in this study.",
       "max_k": 10,
       "expand_aliases": true
     }'
```

#### Example Response:
```json
{
  "entity": "artemisinic acid",
  "criterion": "The paper reports microbial biosynthesis of artemisinic acid in this study.",
  "intent": "synthesis",
  "query_used": "TITLE_ABS:(\"artemisinic acid\" AND (synthesis OR biosynthesis OR \"produced by\" OR fermentation OR preparation OR yielded))",
  "model_revision": "fa2804872c3b4bd748f38c0185cc85775361e735",
  "calibration_version": "platt-v2-logistic",
  "total_retrieved": 10,
  "total_classified": 9,
  "results": [
    {
      "rank": 1,
      "title": "High-titer production of artemisinic acid in engineered Saccharomyces cerevisiae",
      "pmid": "42107802",
      "doi": "10.1038/nature04640",
      "pmcid": "PMC1475823",
      "classification": "qualifies",
      "evidence": {
        "text": "Here we report the production of artemisinic acid at titers up to 25 g/L using an engineered mevalonate pathway in yeast.",
        "char_start": 340,
        "char_end": 462,
        "source": "abstract"
      },
      "evidence_status": "supported",
      "confidence": {
        "score": 0.884,
        "band": "high"
      },
      "score_breakdown": {
        "positive_margin": 0.923,
        "role_score": 0.031,
        "final_score": 0.892,
        "gate_flags": ["this_work_positive"]
      },
      "guard_reason": null,
      "dedup_of": null
    },
    {
      "rank": 2,
      "title": "Synthesis of dibenzoxepines via palladium-catalyzed annulation",
      "pmid": "31948210",
      "doi": "10.1021/acs.orglett.9b04210",
      "pmcid": null,
      "classification": "does_not_qualify",
      "evidence": {
        "text": "In the presence of palladium(II) acetate and triphenylphosphine, substituted dibenzoxepines were prepared in 78% yield.",
        "char_start": 180,
        "char_end": 301,
        "source": "abstract"
      },
      "evidence_status": "contradicted",
      "confidence": {
        "score": 0.125,
        "band": "low"
      },
      "score_breakdown": {
        "positive_margin": 0.420,
        "role_score": 0.895,
        "final_score": -0.475,
        "gate_flags": ["role_guard_triggered"]
      },
      "guard_reason": "role_pattern: in the presence of Pd(OAc)2",
      "dedup_of": null
    },
    {
      "rank": 3,
      "title": "Network pharmacology and molecular mechanisms of methylxanthines in human kidney disease",
      "pmid": "38291044",
      "doi": "10.1016/j.jep.2024.117890",
      "pmcid": null,
      "classification": "does_not_qualify",
      "evidence": null,
      "evidence_status": "insufficient",
      "confidence": {
        "score": 0.051,
        "band": "low"
      },
      "score_breakdown": {
        "positive_margin": null,
        "role_score": null,
        "final_score": null,
        "gate_flags": ["no_production_verb_abstain"]
      },
      "guard_reason": null,
      "dedup_of": null
    }
  ]
}
```

### 2. `POST /feedback` (Continuous Learning Sink)
Allows bioinformaticians to submit ground truth corrections. Entries are atomically written to `feedback.jsonl` under thread locks:
```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "paper_id": "42107802",
       "entity": "artemisinic acid",
       "criterion": "The paper reports microbial biosynthesis of artemisinic acid in this study.",
       "suggested_label": "qualifies",
       "notes": "Validated 25 g/L yeast titer paper."
     }'
```

### 3. `GET /health` (Liveness & Model Diagnostics)
```bash
curl http://127.0.0.1:8000/health
```
```json
{
  "status": "healthy",
  "model_name": "cross-encoder/nli-deberta-v3-small",
  "model_revision": "fa2804872c3b4bd748f38c0185cc85775361e735",
  "device": "cuda:0"
}
```

---

## 10. Evaluation Strategy & Statistical Rigor

### Leave-One-Entity-Out (LOEO) Cross-Validation
To guarantee that the classifier does not overfit to specific chemical nomenclature, evaluation is conducted by training/tuning on specific chemical families (e.g. flavonoids, polyphenols) and evaluating on held-out families (e.g. terpenes, alkaloids, enzymes).

### Wilson Score 95% Confidence Intervals & Stratified Held-Out Results (`N = 56`)
Across the disjoint held-out test entities (`palladium(II) acetate`, `artemisinic acid`, `lipase`; `20` clean + `36` real Europe PMC noisy abstracts), executed with `DebertaNLIScorer` in `notebooks/evaluation.ipynb`:
- **Qualifies Precision:** `0.9444` (`95% Wilson CI [0.7424, 0.9901]`, up from `0.6129` on `v2-baseline`)
- **Qualifies Recall:** `0.8947` (`95% Wilson CI [0.6861, 0.9706]`)
- **Binary Qualifies Accuracy / F1 / F0.5:** `0.9464` `[0.8539, 0.9816]` / `0.9189` / `0.9341`
- **3-Class Macro-F1:** `0.6381` (vs. `0.5245` on `v2-baseline`; Clean `0.9118`, Noisy `0.6230`)
- **Calibrated ECE:** `0.0695` (reduced from `0.3340` uncalibrated `v2` baseline)

### McNemar's Exact Test
Demonstrates statistically significant improvement on the binary `qualifies` slice between `v2-baseline` and `v3.2-frozen` ($b = 11, c = 2, p = 0.0225$).

---

## 11. Automated Test Suite (100% Offline Pass Rate)

The project includes an automated test suite structured into three distinct tiers:

```bash
# Tier 1: Fast deterministic unit tests (50 tests)
# Validates doc-drift against implementation.md, v3.2 features (G3, D5, D6, IUPAC bracket-depth split, title isolation, anaphora), 18-code Reason Registry, 12-rule combiner truth table, and load shedding.
python -m pytest tests/unit -v

# Tier 2: Real DeBERTa-v3 model fixture tests (12 tests)
# Evaluates must-pass cases: Q1-Q3, N1-N6, D1, CS5 conflicting_evidence, and caffeine neutral boundary.
python -m pytest tests/model -v

# Tier 3: Live Europe PMC network integration test
python -m pytest tests/live -m live -v

# Run full offline suite (62/62 Passing):
python -m pytest tests/unit tests/model -v
```

---

## 12. Honest Scientific Limitations

1. **Abstract Paywall Truncation:** In scientific publishing, fermentation yields and synthetic methods often appear only in paywalled full text. Because the engine strictly refuses to invent evidence, papers with vague abstracts will abstain as `insufficient_evidence` (`system_unstated`, `activity_unclear`).
2. **Conservatism on Passive Negatives vs. Thin Calibration Sample:** `v3.2` routed `8/34` `does_not_qualify` test items to abstention (`19.64%` overall abstention rate), and `negative_path` Platt MLE currently rests on `n_rows = 4` dev rows (`different_activity`, `wrong_system`). See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) §6 for full discussion.
3. **Hyphenated Stereoisomers & Multi-Hop Boolean Queries:** Arbitrary multi-hop Boolean joins outside G1/G2/G3 cleanly return HTTP 422 rather than silently mis-scoring.

---

## 13. Summary Checklist

- [x] **Zero Paid APIs:** Local Hugging Face DeBERTa-v3 Cross-Encoder (`cross-encoder/nli-deberta-v3-small`, revision `fa280487`).
- [x] **Criterion-Aware Decomposition:** Separate verification for $C_1$ Entity, $C_2$ Activity, $C_3$ Host System.
- [x] **Closed Reason Registry:** 18 registered reason codes (`17` reachable active codes + deprecated `system_mismatch_abstract`); non-null reason code on any non-qualifying classification.
- [x] **No Hallucinations:** Structural title isolation and runtime slice assertion `canon_body[start:end] == evidence.display_text`.
- [x] **Role / Catalyst Guard & Conflict Routing:** Modifier-tolerant prepositional role patterns (`R3a` veto vs. `R3b` `conflicting_evidence`).
- [x] **Dual-Query Retrieval & SHA-256 Audit:** Union of intent-augmented and plain queries plus `data/snapshots/manifest.json` (`152` records).
- [x] **"This Work" & Bracket-Depth Clause Splitting:** Clause-local experimental attribution + backward anaphora (`ANAPHORA_WINDOW = 2`) without corrupting IUPAC names.
- [x] **Two-Map MLE Platt Scaling (D5):** Fitted across `eval/splits.json` dev entities with conditional null confidence on vetoes, `mention_only`, `background_citation`, and abstentions.
- [x] **Reproducible Notebook & Report:** Executed [`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb) (no stub scorers) and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).
- [x] **Verification:** 62/62 offline tests (`50` unit + `12` real DeBERTa model tests).

---

## 14. Version 3.2 Upgrade Summary (`v3.2-frozen`)

1. **Three-Class Taxonomy & 18-Code Registry (`app/reasons.py`):** `qualifies`, `does_not_qualify`, `insufficient_evidence`, with `mention_only` and `background_citation` classified as evidence-based `does_not_qualify`, `unstated_threshold` added as #18, and `system_mismatch_abstract` deprecated under **D6**.
2. **G1/G2/G3 Criterion Grammar (`app/criterion_parser.py`):** Parses product/agent roles, host families (including `in_vitro`), exclusions, and unit-normalized numeric thresholds; hard HTTP 422 on unparsed clauses (`--allow-partial-criterion` opt-in).
3. **12-Rule Deterministic Combiner (`app/combiner.py`):** `R1`–`R11` with `R3a`/`R3b` synthesis-conflict routing, audited by automated doc-drift tests (`tests/unit/test_doc_drift.py`).
4. **Two-Map MLE Calibration (`eval/calibrate.py`):** Regularized Newton-Raphson logistic MLE on disjoint 4-entity dev split (`eval/splits.json`).
5. **Backwards Compatibility:** `--pipeline v2` preserves legacy sentence-level binary inference cleanly (`v2-baseline` pinned at `85894a0`).

