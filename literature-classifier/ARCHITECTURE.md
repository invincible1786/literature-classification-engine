# Architecture & System Design Specification (`ARCHITECTURE.md`)

## Mini Literature Classification Engine (Loopworm Technical Assessment — v3 Upgrade)

**Candidate:** Ruhaan Kakar  
**Prepared for:** Dr. Jagadeeswara Reddy Devasani, Scientist I – Bioinformatics, Loopworm Private Limited  
**System Scope:** Dual-query Europe PMC retrieval, scoped criterion grammar parsing (`CriterionSpec`), multi-condition verification ($C_1$ Entity, $C_2$ Activity, $C_3$ System), strict precedence combiner, three-class classification (`qualifies`, `does_not_qualify`, `insufficient_evidence`), 17 registered canonical reason codes, verbatim evidence extraction with byte-exact character offsets, Platt-scaled calibrated confidence, curator review queue, and backward compatibility with `--pipeline v2`.  
**Repository Structure:** `loopworm/literature-classifier/`  
**Runtime:** Python 3.11, PyTorch 2.6.0, Transformers, FastAPI, NLTK, AnyIO  

---

## 1. Executive Summary & Design Principles

The Mini Literature Classification Engine solves the core problem defined in the assessment brief: **reliably deciding whether available scientific evidence genuinely supports a user-defined inclusion criterion, rather than merely mentioning the entity**.

### 1.1 First Principles of the Architecture
1. **Zero Paid LLM Dependencies:** Operates strictly on a local, pinned Cross-Encoder (`cross-encoder/nli-deberta-v3-small`, revision `fa280487`) and freely accessible public scientific APIs (Europe PMC REST, PubChem PUG REST).
2. **Criterion-Aware Multi-Condition Verification:** Rather than evaluating monolithic premise-hypothesis pairs, queries are parsed into structured conditions ($C_1$ Entity match & role polarity, $C_2$ Activity qualification & attribution, $C_3$ Host/organism system match).
3. **Strict Verbatim Evidence Coupling:** The document classification score is *strictly* the winning sentence's score. The evidence span returned in the response is guaranteed by an in-code assertion to match the exact slice of canonical text:
   $$\text{canonical\_doc}[\text{char\_start}:\text{char\_end}] == \text{evidence.text}$$
   Titles are never returned as evidence.
4. **Three-Class Taxonomy with Registered Reason Codes:** Documents are mapped to `qualifies`, `does_not_qualify`, or `insufficient_evidence`. Every non-qualifying paper is assigned one of exactly 17 canonical reason codes from the central registry (`app/reasons.py`).
5. **Calibrated, Monotone Uncertainty:** Raw transformer margins ($m_{\text{qualify}}$) are converted via Platt scaling to true calibrated probabilities $P(\text{correct})$, mapped to strictly monotone confidence bands (High $\ge 0.80$, Medium $0.50–0.80$, Low $< 0.50$). Abstaining decisions strictly receive `confidence: null`.
6. **Deliberate Deviations (D1–D4):**
   - **D1:** Single qualifier-path Platt fitting.
   - **D2:** Strict verbatim character offset invariant.
   - **D3:** Hand-set constants registry with LOEO dev tuning and sensitivity analysis (`config/constants.md`).
   - **D4:** Exactly two vetoes (prepositional role pattern veto when `entity_role == "product"` and Europe PMC `pubType=review` when `scope == "this_study"`).
7. **Robust Concurrency & Load Shedding:** Asynchronous `asyncio.Semaphore(2)` is acquired in the router *before* offloading CPU/CUDA tasks to threadpools, guaranteeing polite HTTP 503 load-shedding with `Retry-After: 10` during surges.

---

## 2. End-to-End Pipeline Architecture

```mermaid
flowchart TD
    REQ["POST /classify (entity, criterion, pipeline='v3')"] --> PARSE["[A] Criterion Parser<br/>(Extract G1/G2 grammar, entity_role, activity, system, scope)"]
    PARSE --> RETRIEVE["[B] Europe PMC Dual Query Retrieval<br/>(Query Intent ∪ Query Plain; retrieved_by tag)"]
    RETRIEVE --> DEDUPE["[C] Deduplication & Provenance<br/>(DOI + Title Dedupe; compute source_record_sha256)"]
    DEDUPE --> SEGMENT["[D] Canonicalization & Protected Segmentation<br/>(NFKC, \u2024 masking, Punkt abbreviations)"]
    
    SEGMENT --> C1["[E1] C1: Entity Condition<br/>(Regex lookup, role polarity, related-compound guard)"]
    SEGMENT --> C2["[E2] C2: Activity Condition<br/>(NLI entailment vs negative role, p_this_work attribution)"]
    SEGMENT --> C3["[E3] C3: System Condition<br/>(Gazetteer matching, taxonomy lineage, abstract carve-out)"]
    
    C1 --> COMBINE["[F] Precedence Combiner (Rules 1-11)<br/>(Check Vetoes -> Insufficient Evidence -> Activity / System mismatches -> Qualifies)"]
    C2 --> COMBINE
    C3 --> COMBINE
    
    COMBINE --> EV_ASSERT["[G] Verbatim Evidence Verification<br/>assert canon[start:end] == evidence.display_text<br/>Ensure title != evidence"]
    EV_ASSERT --> CALIB["[H] Platt Scaling Calibration<br/>m_qualify -> P(correct) in [0.0, 1.0]<br/>Null confidence on non-qualifiers"]
    CALIB --> RANK["[I] Deterministic Ranking<br/>1. Qualifiers (by calibrated P desc)<br/>2. Scored Non-Qualifiers<br/>3. Abstained Papers<br/>4. No-Text Papers"]
    
    RANK --> RESP["JSON Response Payload<br/>(api_version='v3', conditions, reason_code, evidence, confidence, source_record_sha256)"]
```

---

## 3. Canonical Reason Registry (`app/reasons.py`)

Every non-qualifying classification is mapped to one of the 17 registered reason codes:

| Reason Code | Label | Kind | Description |
|---|---|---|---|
| `no_text` | `insufficient_evidence` | `abstain` | Neither abstract nor full-text available for paper. |
| `abstract_only` | `insufficient_evidence` | `abstain` | Paper has abstract only; conclusive qualification requires full-text. |
| `mention_only` | `does_not_qualify` | `evidence-based` | Entity mentioned without satisfying production or activity verbs. |
| `activity_unclear` | `insufficient_evidence` | `abstain` | Activity mentioned with ambiguous or insufficient experimental evidence. |
| `entity_unclear` | `insufficient_evidence` | `abstain` | Entity mention is ambiguous or truncated. |
| `entity_absent` | `does_not_qualify` | `evidence-based` | Entity absent in scored full text. Carve-out: never emitted for abstract-only. |
| `system_unstated` | `insufficient_evidence` | `abstain` | Target organism or host system not stated in available text. |
| `system_mismatch_abstract` | `insufficient_evidence` | `abstain` | Different organism detected in abstract; host system may appear in full-text. |
| `conflicting_evidence` | `insufficient_evidence` | `abstain` | Sentences present conflicting entailment and contradiction signals. |
| `evidence_split` | `insufficient_evidence` | `abstain` | Activity and entity evidence separated across non-contiguous sentences. |
| `unsupported_criterion_clause` | `insufficient_evidence` | `abstain` | User criterion includes clauses outside supported grammar (G1/G2). |
| `different_activity` | `does_not_qualify` | `evidence-based` | Paper describes different activity (e.g. pharmacology instead of synthesis). |
| `wrong_system` | `does_not_qualify` | `evidence-based` | Explicit conflicting organism confirmed in full text. |
| `reagent_role` | `does_not_qualify` | `veto` | Entity acts as reagent, catalyst, or substrate, not the synthesized product. |
| `background_citation` | `does_not_qualify` | `evidence-based` | Entity/activity cited strictly as background or prior literature. |
| `possible_review` | `insufficient_evidence` | `abstain` | Review-style language detected in abstract without pubType metadata confirmation. |
| `review_article` | `does_not_qualify` | `veto` | Europe PMC pubType confirms paper is a Review / Systematic Review. |

---

## 4. Multi-Condition Verification & Precedence Rules

### 4.1 Condition Checkers
1. **$C_1$ (Entity Match & Role Polarity):**
   - Matches entity or vetted aliases via regex boundaries.
   - Asserts role polarity: if `entity_role == "product"`, checks prepositional patterns (`config/role_patterns.json`). If entity is preceded by `in the presence of`, `catalyzed by`, etc., flags reagent role.
   - Enforces related-compound guard (e.g., blocks `"artemisinin"` when seeking `"artemisinic acid"`).
2. **$C_2$ (Activity Qualification & Attribution):**
   - Runs cross-encoder NLI scoring over positive activity templates vs negative role hypothesis.
   - Computes $p_{\text{this\_work}}$ attribution to disqualify background citations.
3. **$C_3$ (Host / Organism System):**
   - Uses organism gazetteer (`config/systems.json`) to detect organism family (`microbial`, `yeast`, `bacteria`, `fungi`, `plant`, `mammalian`).
   - Carve-out: if paper is abstract-only and organism is unstated, assigns `system_unstated` (or `system_mismatch_abstract` if an alternative organism is mentioned) rather than hard `wrong_system`.

### 4.2 Combiner Precedence Rules (1–11)
When combining $C_1, C_2, C_3$, the combiner executes strict precedence:
1. `no_text`: Paper text is empty $\rightarrow$ `insufficient_evidence`.
2. `review_article`: Europe PMC `pubType=review` $\rightarrow$ `does_not_qualify` (`reasons.review_article`).
3. `reagent_role`: Entity role veto triggered $\rightarrow$ `does_not_qualify` (`reasons.reagent_role`).
4. `entity_absent`: Scored text lacks entity (full-text only) $\rightarrow$ `does_not_qualify`.
5. `unsupported_criterion_clause`: Criterion contains unsupported clause $\rightarrow$ `insufficient_evidence`.
6. `background_citation`: Activity is background citation $\rightarrow$ `does_not_qualify`.
7. `system_mismatch_abstract`: Organism mismatch in abstract-only paper $\rightarrow$ `insufficient_evidence`.
8. `wrong_system`: Confirmed wrong organism in full-text $\rightarrow$ `does_not_qualify`.
9. `system_unstated`: Required system absent $\rightarrow$ `insufficient_evidence`.
10. `different_activity`: Non-qualifying activity $\rightarrow$ `does_not_qualify`.
11. `qualifies`: All active conditions satisfied $\rightarrow$ `qualifies`.

---

## 5. Concurrency, Provenance & Load Shedding

1. **FastAPI Concurrency Slot Limiter:**
   In `app/main.py`, incoming requests acquire an `asyncio.Semaphore(2)` before entering handlers:
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
2. **Dual-Query Retrieval & Deduplication:**
   `app/retrieval.py` unions the intent-augmented query with plain `TITLE_ABS:"{entity}"`, tagging each retrieved record with `retrieved_by: "intent" | "plain" | "both"`.
3. **SHA-256 Provenance Tracking:**
   Every candidate record calculates `source_record_sha256 = SHA256(title + " " + abstractText)` to guarantee exact provenance and reproducible offline evaluation.
4. **Curator Feedback Queue:**
   `POST /feedback` atomically logs human corrections to `feedback.jsonl` under thread lock protection, validated through `eval/ingest_feedback.py`.

---

## 6. Verification & Test Suite Summary

The engine is protected by a 35-test multi-tier automated test suite:
- **Tier 1 (35 Unit Tests):** Verifies Reason Registry, Criterion Parser, Condition Checkers, Role Polarity, Combiner Precedence Truth Table, Abstract Carve-Outs, Schema Invariants, Regex lookarounds, and Monotone Bands. (Runs in ~12.6 seconds, 35/35 Green).
- **Tier 2 (10 Model Fixture Tests):** Evaluates real DeBERTa-v3 inference on benchmark target sentences (Q1–Q3, N1–N6, D1).
- **Tier 3 (Live Tests):** Validates live Europe PMC dual retrieval and PubChem PUG REST integration.
- **Pass Rate:** **100% Pass Rate**.
