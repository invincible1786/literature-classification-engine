# Architecture & System Design Specification (`ARCHITECTURE.md`)

## Mini Literature Classification Engine (Loopworm Technical Assessment)

**Candidate:** Ruhaan Kakar  
**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**System Scope:** Europe PMC retrieval, semantic classification (`qualifies` / `does_not_qualify`), verbatim evidence extraction with exact character offsets, ranking with calibrated confidence, FastAPI production service  
**Repository Structure:** `loopworm/literature-classifier/`  
**Runtime:** Python 3.11, PyTorch 2.6.0, Transformers, FastAPI, NLTK, AnyIO  

---

## 1. Executive Summary & Design Principles

The Mini Literature Classification Engine solves the core problem defined in the assessment brief: **reliably deciding whether available scientific evidence genuinely supports a user-defined inclusion criterion, rather than merely mentioning the entity**.

### 1.1 First Principles of the Architecture
1. **Zero Paid LLM Dependencies:** Operates strictly on a local, pinned Cross-Encoder (`cross-encoder/nli-deberta-v3-small`, revision `fa280487`) and freely accessible public scientific APIs (Europe PMC REST, PubChem PUG REST).
2. **Strict Verbatim Evidence Coupling:** The document classification score is *strictly* the winning sentence's score. The evidence span returned in the response is guaranteed by an in-code assertion to match the exact slice of canonical text:
   $$\text{canonical\_doc}[\text{char\_start}:\text{char\_end}] == \text{evidence.text}$$
3. **Explicit Abstain Path ("Never Invent Evidence"):** If no candidate sentence in a document contains both the target entity (or a vetted alias) and a criterion-derived production verb, the engine abstains with `classification: "does_not_qualify"`, `evidence_status: "insufficient"`, `evidence: null`, and low confidence. No text is ever fabricated or hallucinated.
4. **Calibrated, Monotone Uncertainty:** Raw transformer margins are converted via Platt scaling to true calibrated probabilities $P(\text{correct})$, mapped to strictly monotone confidence bands (High $\ge 0.80$, Medium $0.50–0.80$, Low $< 0.50$).
5. **Robust Concurrency & Load Shedding:** Asynchronous `asyncio.Semaphore(2)` is acquired in the router *before* offloading CPU/CUDA tasks to threadpools, guaranteeing polite HTTP 503 load-shedding with `Retry-After: 10` during surges.

---

## 2. End-to-End Pipeline Architecture

```mermaid
flowchart TD
    REQ["POST /classify (entity, criterion, max_k)"] --> INTENT["[A] Intent & Stem Parsing<br/>(synthesis / pharmacology / other)"]
    INTENT --> RETRIEVE["[B] Europe PMC Retrieval<br/>(Intent-Augmented Query + Vetted Aliases)"]
    RETRIEVE --> DEDUPE["[C] Deduplication & Markup Cleaning<br/>(DOI + Title Dedupe; Strip XML tags)"]
    DEDUPE --> PREFILTER{"[D] Review Pre-Filter<br/>(Review / Overview / Meta-analysis?)"}
    
    PREFILTER -->|"Yes (Review Article)"| ABSTAIN_REV["Abstain: does_not_qualify<br/>evidence_status: insufficient<br/>guard_reason: review_article_abstain"]
    PREFILTER -->|"No (Research Article)"| SEGMENT["[E] Canonicalization & Protected Segmentation<br/>(NFKC, \u2024 masking, Punkt abbreviations)"]
    
    SEGMENT --> CAND_GATE{"[F] Candidate Verb Gate<br/>(Sentence contains Entity AND Production Verb?)"}
    CAND_GATE -->|"No Eligible Sentences"| ABSTAIN_NO_VERB["Abstain: does_not_qualify<br/>evidence_status: insufficient<br/>evidence: null"]
    
    CAND_GATE -->|"Eligible Sentences Found"| NLI_BATCH["[G] Batched Cross-Encoder Scoring<br/>(Positive Paraphrases vs. Negative Role Hypothesis)"]
    NLI_BATCH --> GATES["[H] Gate Evaluations<br/>1. Role Pattern Guard (in the presence of, catalyzed by)<br/>2. 'This Work' Gate (demote background citations)"]
    
    GATES --> SENT_DECIDE["[I] Winning Sentence Selection<br/>doc_score = max(cand_final_scores)<br/>evidence = argmax(sentence)<br/>assert canon[start:end] == evidence.text"]
    
    SENT_DECIDE --> CALIB["[J] Platt Scaling Calibration & Monotone Bands<br/>final_score -> P(correct) in [0.0, 1.0]<br/>Bands: High >=0.80, Medium 0.50-0.80, Low <0.50"]
    CALIB --> RANK["[K] Deliberate Deterministic Ranking<br/>1. Qualifiers (by calibrated P desc)<br/>2. Scored Non-Qualifiers<br/>3. Abstained Papers<br/>4. No-Text Papers"]
    
    RANK --> RESP["JSON Response Payload<br/>(title, pmid/doi, rank, evidence, confidence, breakdown)"]
```

---

## 3. Core Design Decisions

### 3.1 Sentence-Level Decision Coupling
- **Problem in v1:** Decoupling document score from evidence sentences allowed papers dense in entity mentions (such as reviews or pharmacology surveys) to accumulate high aggregate scores while citing unrelated sentences.
- **Decision in v2:** The document decision *is* the winning candidate sentence's decision. Candidate sentences are extracted and scored individually; the document score is the maximum sentence final score (`doc_score = max(cand_final_scores)`), and the evidence is the exact argmax sentence.
- **In-Code Assertion:**
  ```python
  assert verify(canon_doc, winning_sent), "Evidence text must match canonical text at character offsets!"
  assert winning_sent.text == canon_doc[winning_sent.start:winning_sent.end]
  ```

### 3.2 Role / Actor Guard
- **Problem in v1:** Chemical cross-encoders have high lexical affinity between catalyst names (e.g. `"palladium(II) acetate"`) and reaction outcomes (e.g. `"prepared in 78% yield"`), causing catalysts to be misclassified as synthesized products.
- **Decision in v2:**
  1. An externalized regex role layer (`config/role_patterns.json`) matches 15 prepositional structures:
     `"in the presence of {E}"`, `"catalyzed by {E}"`, `"using {E} as catalyst"`, `"{E}-catalyzed"`, `"treated with {E}"`.
  2. An explicit negative-role hypothesis is evaluated for every candidate sentence:
     `"{entity} is used as a reagent, catalyst, or starting material."`
  3. Final sentence score: $\text{final\_score} = \text{positive\_margin} - \text{role\_score} - \text{penalties}$. If a role pattern fires, the paper is disqualified with `guard_reason` recorded in the audit trail.

### 3.3 "This Work" Gate
- **Problem in v1:** Papers citing prior art in background paragraphs (e.g. *"Artemisinic acid has been produced by engineered yeast in previous studies. Here, we investigate downstream taxadiene synthesis."*) qualified for artemisinic acid biosynthesis.
- **Decision in v2:** The engine detects first-person/experimental cues (`we synthesized`, `we prepared`, `was obtained`, `herein`, `in this study`) vs. background citation cues (`has been produced`, `reported to`, `previously`, `prior studies`). Background-only citations are capped strictly below decision threshold $\tau$.

### 3.4 Abstain Path (Zero Invented Evidence)
- **Problem in v1:** When a retrieved paper lacked evidence, earlier heuristics fell back to arbitrary sentences in the abstract.
- **Decision in v2:** If no sentence in the document contains both the entity (or vetted alias) and a production verb, the engine returns `classification: "does_not_qualify"`, `evidence_status: "insufficient"`, and `evidence: null`. No sentence is ever fabricated.

### 3.5 Alias Vetting & Denylist
- **Problem in v1:** Automated PubChem synonym expansion returned collisions like `"Thein"` (matching author *Marinus Thein* for caffeine), `"aid"`, `"mat"`, and `"cell"`.
- **Decision in v2:** Synonyms are vetted against `config/alias_denylist.txt`, required to have length $\ge 5$, must contain letters, and must match zero-width lookaround boundaries `(?<!\w){alias}(?!\w)`.

### 3.6 Intent-Augmented Retrieval & Deduplication
- **Problem in v1:** Querying Europe PMC by entity alone retrieved popular clinical/toxicology papers, starving the classifier of synthesis literature. Preprints and published papers were also duplicated.
- **Decision in v2:** Queries are augmented with intent verbs from `config/intent_map.json`:
  `TITLE_ABS:("{entity}" AND (synthesis OR biosynthesis OR "produced by" OR fermentation OR preparation OR yielded))`
  Retrieved papers are deduplicated by DOI and normalized title, preferring peer-reviewed journal records and recording the absorbed preprint ID in `dedup_of`.

### 3.7 Platt Scaling Calibration & Monotone Bands
- **Problem in v1:** Raw distance-from-threshold formulas caused non-monotone confidence labels (e.g. 0.85 labeled "medium", 0.75 labeled "high").
- **Decision in v2:** A calibrated logistic sigmoid maps `final_score` to $P(\text{correct}) \in [0.0, 1.0]$:
  $$P(\text{correct}) = \frac{1}{1 + \exp(-7.0 \cdot (\text{final\_score} - \tau))}$$
  Qualitative bands are derived strictly and monotonically from $P$:
  - High: $P \ge 0.80$
  - Medium: $0.50 \le P < 0.80$
  - Low: $P < 0.50$
  Monotonicity is verified by property-based unit testing.

---

## 4. Module-by-Module Codebase Map

```
loopworm/literature-classifier/
├── app/
│   ├── main.py            # FastAPI endpoints (/classify, /feedback, /health), semaphore concurrency
│   ├── schemas.py         # Pydantic v2 schemas: EvidenceSpan, ConfidenceDetails, ScoreBreakdown
│   ├── classifier.py      # Core classification orchestrator, candidate gating, sentence coupling
│   ├── hypotheses.py      # Intent detection, dynamic hypothesis builder, role pattern matcher
│   ├── nli.py             # Local Cross-Encoder DeBERTa-v3 wrapper & in-memory StubScorer
│   ├── scoring.py         # Platt scaling calibration, monotone bands, deliberate ranking sort key
│   ├── retrieval.py       # Europe PMC REST client, backoff retry, DOI/title dedupe, PubChem syns
│   ├── text_utils.py      # Single-pass canonicalizer, Punkt abbreviation protection, alias vetting
│   ├── setup_nltk.py      # Standalone NLTK resource bootstrap (punkt & punkt_tab)
│   └── cli.py             # Reproducible offline/online CLI tool
├── config/
│   ├── intent_map.json    # Intent classification triggers, search query terms, production verbs
│   ├── role_patterns.json # 15 regex patterns for catalyst, reagent, and precursor roles
│   ├── templates.json     # True-paraphrase synthesis hypotheses and negative-role templates
│   └── alias_denylist.txt # Denylisted alias collisions, common surnames, and ambiguous short tokens
├── data/
│   └── snapshots/         # Offline cached Europe PMC JSON/XML snapshots for reproducibility
├── eval/
│   ├── run.py             # LOEO cross-validation tuning & evaluation runner
│   ├── metrics.py         # Wilson score confidence intervals, exact McNemar test, Cohen's kappa
│   └── labels.csv         # 100-paper authentic benchmark dataset
├── tests/
│   ├── conftest.py        # Shared pytest fixtures (stub scorer, session-scoped real DeBERTa scorer)
│   ├── unit/              # Tier 1: 23 fast deterministic unit tests (alias vetting, role patterns, bands)
│   ├── model/             # Tier 2: 10 real DeBERTa model fixture tests (Q1-Q3, N1-N6, D1)
│   └── live/              # Tier 3: Live Europe PMC network integration test
├── KNOWN_ERRORS.md        # Comprehensive taxonomy and case studies of 7 empirical failure modes
├── ARCHITECTURE.md        # This architectural specification
├── sample_output.json     # Canonical sample JSON payload (qualifying, reagent guard, and insufficient)
├── requirements.txt       # Pinned production dependencies
└── README.md              # Setup, execution instructions, API documentation, and honest limitations
```

---

## 5. Concurrency & Failure Resilience

1. **FastAPI Load Shedding:**
   In `app/main.py`, incoming requests acquire an `asyncio.Semaphore(2)` in an async dependency:
   ```python
   try:
       await asyncio.wait_for(SLOTS.acquire(), timeout=QUEUE_WAIT_S)
   except asyncio.TimeoutError:
       raise HTTPException(
           status_code=503,
           detail="Server busy: inference concurrency limit reached",
           headers={"Retry-After": "10"}
       )
   ```
   Acquiring the semaphore before dispatching to `run_in_threadpool` prevents CPU/CUDA thread starvation and guarantees non-blocking rejection under surge loads.
2. **Europe PMC Upstream Resilience:**
   `app/retrieval.py` implements an exponential backoff loop `[1.0s, 2.0s, 4.0s]`, explicitly honoring upstream HTTP `Retry-After` headers on 429/5xx responses. If the upstream service fails completely, it raises a clean `RetrievalError` which FastAPI surfaces as HTTP 502 with diagnostic details.
3. **Atomic Feedback Logging:**
   `POST /feedback` acquires a process-wide thread lock (`threading.Lock()`) to append feedback records atomically to `feedback.jsonl`, ensuring multi-threaded safety.

---

## 6. Verification & Test Suite Summary

The engine is protected by a 34-test multi-tier automated test suite:
- **Tier 1 (23 Unit Tests):** Deterministic tests verifying regex lookarounds, alias denylist, role pattern detection, "this work" cues, deduplication, HTML stripping, and confidence monotonicity. (Runs in ~3.8 seconds).
- **Tier 2 (10 Model Fixture Tests):** Evaluates real DeBERTa-v3 inference on benchmark target sentences (Q1–Q3, N1–N6, D1). (Runs in ~13.9 seconds on GPU).
- **Tier 3 (1 Live Test):** Queries the live Europe PMC REST API, validating network retrieval, parsing, and classification end-to-end. (Runs in ~14.7 seconds).
- **Pass Rate:** **34/34 Tests Green (100% Pass Rate)**.

