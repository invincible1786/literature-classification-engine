"""FastAPI application providing /classify, /feedback, and /health endpoints."""
import os
import time
import json
import asyncio
import logging
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.concurrency import run_in_threadpool

from app.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    PaperClassificationResult,
    FeedbackRequest,
    EvidenceSpan,
    ConfidenceDetails,
    ScoreBreakdown,
)
from app.retrieval import (
    search_europe_pmc,
    search_europe_pmc_dual,
    fetch_pubchem_synonyms,
    deduplicate_papers,
    RetrievalError,
)
from app.text_utils import init_punkt, vet_aliases
from app.hypotheses import detect_intent
from app.nli import DebertaNLIScorer, StubScorer, NLIScorer, DEFAULT_MODEL_NAME, DEFAULT_REVISION
from app.classifier import (
    classify_publications,
    DEFAULT_THRESHOLD,
    DEFAULT_CONTRAST_DELTA,
)
from app.criterion_parser import (
    CriterionSpec,
    CriterionParseError,
    parse_criterion_to_spec,
)
from app.scoring import CALIBRATION_VERSION, load_calibration_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("literature-classifier")

# Global scorer instance
scorer: Optional[NLIScorer] = None

# Concurrency and Load Control
MAX_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
SLOTS = asyncio.Semaphore(MAX_JOBS)
QUEUE_WAIT_S = float(os.getenv("QUEUE_WAIT_S", "5"))

# Process-wide lock for safe atomic feedback logging
_FEEDBACK_LOCK = threading.Lock()

# In-memory response cache (TTL 1 hour)
_RESPONSE_CACHE: Dict[str, Tuple[float, ClassifyResponse]] = {}
_CACHE_TTL_S = 3600.0


def build_lucene_query(entity: str, aliases: Optional[list] = None, intent: str = "synthesis") -> str:
    terms = [f'"{entity}"']
    if aliases:
        for a in aliases[:3]:
            if a.lower() != entity.lower():
                terms.append(f'"{a}"')
    entity_part = " OR ".join(terms)
    if intent == "synthesis":
        return f'TITLE_ABS:(({entity_part}) AND (synthesis OR biosynthesis OR "produced by" OR fermentation OR preparation OR yielded))'
    elif intent == "pharmacology":
        return f'TITLE_ABS:(({entity_part}) AND (pharmacology OR receptor OR pharmacokinetic OR mechanism OR clinical))'
    return f'TITLE_ABS:({entity_part})'


async def slot_dependency():
    """Acquires a concurrency slot before entering the handler; sheds load politely with 503."""
    try:
        await asyncio.wait_for(SLOTS.acquire(), timeout=QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server busy: inference concurrency limit reached",
            headers={"Retry-After": "10"}
        )
    try:
        yield
    finally:
        SLOTS.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan warmup: initializes NLTK punkt and pre-warms the NLI CrossEncoder."""
    global scorer
    logger.info("Initializing NLP components and pre-warming...")
    init_punkt()
    load_calibration_config()

    if os.getenv("USE_STUB_SCORER", "0") == "1":
        logger.info("TEST_MODE active: using in-memory StubScorer")
        scorer = StubScorer()
    else:
        scorer = DebertaNLIScorer()

    app.state.scorer = scorer
    logger.info("Lifespan warmup complete. Ready to serve traffic.")
    yield
    logger.info("Shutting down literature classification engine.")


app = FastAPI(
    title="Mini Literature Classification Engine",
    description="Deterministic literature mining and semantic classification engine for Loopworm (v3).",
    version="3.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    """Service health check and model provenance."""
    active_scorer = getattr(app.state, "scorer", scorer)
    device = getattr(active_scorer, "device", "cpu") if active_scorer else "uninitialized"
    revision = getattr(active_scorer, "revision", DEFAULT_REVISION) if active_scorer else DEFAULT_REVISION
    model_name = getattr(active_scorer, "model_name", DEFAULT_MODEL_NAME) if active_scorer else DEFAULT_MODEL_NAME
    return {
        "status": "healthy",
        "model_name": model_name,
        "model_revision": revision,
        "calibration_version": CALIBRATION_VERSION,
        "pipeline": "v3.2",
        "device": str(device)
    }


def _run_pipeline(request: ClassifyRequest) -> ClassifyResponse:
    """Synchronous CPU/CUDA execution routine dispatched to AnyIO worker threadpool."""
    active_scorer = getattr(app.state, "scorer", scorer)
    if active_scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLI model is not yet initialized."
        )

    start_time = time.perf_counter()

    # 1. Alias handling with PubChem expansion and denylist vetting
    raw_aliases = list(request.aliases) if request.aliases else []
    if request.expand_aliases and not request.offline:
        pubchem_syns = fetch_pubchem_synonyms(request.entity)
        for syn in pubchem_syns:
            if syn not in raw_aliases and len(raw_aliases) < 8:
                raw_aliases.append(syn)

    vetted_aliases = vet_aliases(raw_aliases, request.entity)

    # 2. Intent detection and criterion parsing
    intent = detect_intent(request.criterion, request.entity)
    query_intent = build_lucene_query(request.entity, vetted_aliases, intent=intent)
    query_plain = f'TITLE_ABS:"{request.entity}"'

    # Validate criterion grammar or catch unparsed clauses
    criterion_spec = request.criterion_spec
    if criterion_spec is None:
        try:
            criterion_spec = parse_criterion_to_spec(
                criterion=request.criterion,
                entity=request.entity,
                allow_partial_criterion=request.allow_partial_criterion
            )
        except CriterionParseError as err:
            logger.warning(f"Criterion parse error: {err}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": err.message,
                    "unparsed_clauses": err.unparsed_clauses,
                    "supported_grammars": [
                        "G1: <paper> reports <activity> of <entity> [in <system>] [in this study]",
                        "G2: <paper> uses <entity> as a <catalyst|reagent|substrate> [in <activity>]"
                    ]
                }
            )

    # 3. Check response cache
    cache_key = f"{request.entity.lower()}:{request.criterion.lower()}:{request.max_results}:{request.threshold}:{request.pipeline}:{getattr(active_scorer, 'revision', 'v3')}"
    now = time.time()
    if cache_key in _RESPONSE_CACHE:
        cached_ts, cached_resp = _RESPONSE_CACHE[cache_key]
        if now - cached_ts < _CACHE_TTL_S:
            logger.info(f"Returning cached response for entity '{request.entity}'")
            return cached_resp

    logger.info(f"Retrieving publications: entity='{request.entity}', intent='{intent}', pipeline={request.pipeline}")

    # 4. Search Europe PMC (Dual Query in v3)
    t_retrieval_start = time.perf_counter()
    try:
        if request.pipeline == "v3":
            publications, dedup_count = search_europe_pmc_dual(
                query_intent=query_intent,
                query_plain=query_plain,
                max_results=request.max_results,
                include_fulltext=request.include_fulltext,
                offline_fallback_entity=request.entity if request.offline else None,
                search_fn=search_europe_pmc
            )
            raw_publications = publications
        else:
            raw_publications = search_europe_pmc(
                query=query_intent,
                max_results=request.max_results,
                include_fulltext=request.include_fulltext,
                offline_fallback_entity=request.entity if request.offline else None
            )
            publications, dedup_count = deduplicate_papers(raw_publications)
    except RetrievalError as err:
        logger.error(f"Retrieval error from Europe PMC: {err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Europe PMC service error: {err}"
        )
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    # 5. Handle zero results gracefully
    if not publications:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ClassifyResponse(
            api_version="v3",
            entity=request.entity,
            query_entity=request.entity,
            criterion=request.criterion,
            criterion_spec=criterion_spec,
            intent=intent,
            query_used=query_intent,
            query_plain=query_plain,
            entity_aliases_used=vetted_aliases,
            aliases_used=vetted_aliases,
            n_retrieved=len(raw_publications),
            total_retrieved=len(raw_publications),
            n_after_dedupe=0,
            total_classified=0,
            dedup_count=dedup_count,
            retrieved_by="both",
            execution_time_ms=round(elapsed_ms, 2),
            timings_ms={"retrieval_ms": round(elapsed_ms, 2)},
            model_revision=getattr(active_scorer, "revision", DEFAULT_REVISION),
            calibration_version=CALIBRATION_VERSION,
            ensemble_size=1,
            hypotheses_used=[request.criterion],
            contrast_guard_active=(intent == "synthesis"),
            guard_arming={"intent": intent, "spec": criterion_spec.model_dump() if hasattr(criterion_spec, "model_dump") else (criterion_spec.dict() if criterion_spec else {})},
            results=[]
        )

    # 6. Execute classification
    run_output = classify_publications(
        publications=publications,
        entity=request.entity,
        criterion=request.criterion,
        aliases=vetted_aliases,
        tau=request.threshold or DEFAULT_THRESHOLD,
        delta=request.contrast_delta or DEFAULT_CONTRAST_DELTA,
        custom_contrast_hypotheses=request.custom_contrast_hypotheses,
        user_paraphrases=request.user_paraphrases,
        scorer=active_scorer,
        pipeline=request.pipeline,
        abstract_negative_policy=request.abstract_negative_policy,
        allow_partial_criterion=request.allow_partial_criterion,
        criterion_spec=criterion_spec
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    pydantic_results: List[PaperClassificationResult] = []
    for r in run_output.results:
        ev_obj = r.get("evidence")
        if ev_obj and isinstance(ev_obj, dict):
            ev_obj = EvidenceSpan(**ev_obj)

        conf_obj = r.get("confidence")
        if conf_obj and isinstance(conf_obj, dict):
            conf_obj = ConfidenceDetails(**conf_obj)
        elif isinstance(conf_obj, (int, float)):
            conf_score = float(conf_obj)
            band = "High" if conf_score >= 0.80 else ("Medium" if conf_score >= 0.50 else "Low")
            conf_obj = ConfidenceDetails(score=conf_score, value=conf_score, band=band)

        sb_obj = r.get("score_breakdown")
        if sb_obj and isinstance(sb_obj, dict):
            sb_obj = ScoreBreakdown(**sb_obj)

        pydantic_results.append(
            PaperClassificationResult(
                api_version=r.get("api_version", "v3"),
                paper_id=str(r["paper_id"]),
                rank=r.get("rank", 1),
                title=r.get("title"),
                doi=r.get("doi"),
                pmid=r.get("pmid"),
                pmcid=r.get("pmcid"),
                pub_year=str(r.get("pub_year")) if r.get("pub_year") else None,
                classification=r["classification"],
                classification_reason=r.get("classification_reason", r["classification"]),
                reason_code=r.get("reason_code"),
                criterion_spec=criterion_spec,
                evidence=ev_obj,
                evidence_sentence=r.get("evidence_sentence"),
                evidence_status=r["evidence_status"],
                evidence_source=r.get("evidence_source", "abstract_only"),
                evidence_offsets=r.get("evidence_offsets"),
                source_record_sha256=r.get("source_record_sha256"),
                text_sha256=r.get("text_sha256"),
                aliases_used=vetted_aliases,
                retrieved_by=r.get("retrieved_by", "intent"),
                conditions=r.get("conditions"),
                qualify_score=r.get("qualify_score"),
                confidence=conf_obj,
                confidence_band=r.get("confidence_band", "Low"),
                confidence_details=conf_obj,
                confidence_kind=r.get("confidence_kind", "calibrated_platt_probability"),
                score_breakdown=sb_obj,
                guard_reason=r.get("guard_reason"),
                dedup_of=r.get("dedup_of"),
                k_eligible=r.get("k_eligible", 0),
                k_scored=r.get("k_scored", 0),
                reagent_signal=r.get("reagent_signal", 0.0),
                timings_ms=r.get("timings_ms", {})
            )
        )

    response = ClassifyResponse(
        api_version="v3",
        entity=request.entity,
        query_entity=request.entity,
        criterion=request.criterion,
        criterion_spec=criterion_spec,
        intent=intent,
        query_used=query_intent,
        query_plain=query_plain,
        entity_aliases_used=vetted_aliases,
        aliases_used=vetted_aliases,
        n_retrieved=len(raw_publications),
        total_retrieved=len(raw_publications),
        n_after_dedupe=len(publications),
        total_classified=len(pydantic_results),
        dedup_count=dedup_count,
        retrieved_by="both",
        execution_time_ms=round(elapsed_ms, 2),
        timings_ms={
            **(getattr(run_output, "timings_ms", {}) or {}),
            "retrieval_ms": round(retrieval_ms, 2),
            "total_execution_ms": round(elapsed_ms, 2),
        },
        model_revision=getattr(active_scorer, "revision", DEFAULT_REVISION),
        calibration_version=CALIBRATION_VERSION,
        ensemble_size=len(run_output.hypotheses_used),
        hypotheses_used=run_output.hypotheses_used,
        contrast_guard_active=run_output.contrast_guard_active,
        guard_arming=run_output.guard_arming,
        results=pydantic_results
    )

    _RESPONSE_CACHE[cache_key] = (now, response)
    return response


@app.post("/classify", response_model=ClassifyResponse)
@app.post("/search-and-classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest, _=Depends(slot_dependency)):
    """
    Main literature classification endpoint.
    Retrieves candidate publications, parses CriterionSpec, applies multi-condition checks,
    and runs cross-encoder NLI inference offloaded to AnyIO worker threadpools.
    """
    return await run_in_threadpool(_run_pipeline, request)


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    Atomically records user corrections to feedback.jsonl with thread safety.
    Acts as a curator review queue.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    feedback_file = os.path.join(base_dir, "feedback.jsonl")

    active_scorer = getattr(app.state, "scorer", scorer)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_revision": getattr(active_scorer, "revision", DEFAULT_REVISION),
        "paper_id": feedback.paper_id,
        "entity": feedback.entity,
        "criterion": feedback.criterion,
        "suggested_label": feedback.suggested_label,
        "notes": feedback.notes
    }

    try:
        with _FEEDBACK_LOCK:
            with open(feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return {"status": "success", "message": f"Feedback recorded for {feedback.paper_id}"}
    except Exception as e:
        logger.error(f"Failed to append feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not record feedback: {e}"
        )
