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
    build_query,
    search_europe_pmc,
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
from app.scoring import CALIBRATION_VERSION

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
    description="Deterministic literature mining and semantic classification engine for Loopworm.",
    version="2.0.0",
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

    # 2. Intent detection and intent-augmented Lucene query construction
    intent = detect_intent(request.criterion, request.entity)
    query = build_query(request.entity, vetted_aliases, intent=intent)

    # 3. Check response cache
    cache_key = f"{request.entity.lower()}:{request.criterion.lower()}:{request.max_results}:{request.threshold}:{request.contrast_delta}:{getattr(active_scorer, 'revision', 'v2')}"
    now = time.time()
    if cache_key in _RESPONSE_CACHE:
        cached_ts, cached_resp = _RESPONSE_CACHE[cache_key]
        if now - cached_ts < _CACHE_TTL_S:
            logger.info(f"Returning cached response for entity '{request.entity}'")
            return cached_resp

    logger.info(f"Retrieving publications: entity='{request.entity}', intent='{intent}', query='{query}', max_results={request.max_results}")

    # 4. Search Europe PMC (or load offline snapshot)
    try:
        raw_publications = search_europe_pmc(
            query=query,
            max_results=request.max_results,
            include_fulltext=request.include_fulltext,
            offline_fallback_entity=request.entity if request.offline else None
        )
    except RetrievalError as err:
        logger.error(f"Retrieval error from Europe PMC: {err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Europe PMC service error: {err}"
        )

    # 5. Deduplicate publications (prefer published over preprints)
    publications, dedup_count = deduplicate_papers(raw_publications)

    # 6. Handle zero results gracefully
    if not publications:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ClassifyResponse(
            entity=request.entity,
            query_entity=request.entity,
            criterion=request.criterion,
            intent=intent,
            query_used=query,
            entity_aliases_used=vetted_aliases,
            n_retrieved=len(raw_publications),
            total_retrieved=len(raw_publications),
            n_after_dedupe=0,
            total_classified=0,
            dedup_count=dedup_count,
            execution_time_ms=round(elapsed_ms, 2),
            model_revision=getattr(active_scorer, "revision", DEFAULT_REVISION),
            calibration_version=CALIBRATION_VERSION,
            ensemble_size=1,
            hypotheses_used=[request.criterion],
            contrast_guard_active=(intent == "synthesis"),
            guard_arming={"intent": intent, "armed": (intent == "synthesis")},
            results=[]
        )

    # 7. Execute semantic classification & sentence coupling
    run_output = classify_publications(
        publications=publications,
        entity=request.entity,
        criterion=request.criterion,
        aliases=vetted_aliases,
        threshold=request.threshold,
        contrast_delta=request.contrast_delta,
        custom_contrast_hypotheses=request.custom_contrast_hypotheses,
        user_paraphrases=request.user_paraphrases,
        scorer=active_scorer
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
            band = "high" if conf_score >= 0.80 else ("medium" if conf_score >= 0.50 else "low")
            conf_obj = ConfidenceDetails(score=conf_score, band=band)

        sb_obj = r.get("score_breakdown")
        if sb_obj and isinstance(sb_obj, dict):
            sb_obj = ScoreBreakdown(**sb_obj)

        pydantic_results.append(
            PaperClassificationResult(
                paper_id=str(r["paper_id"]),
                rank=r.get("rank", 1),
                title=r.get("title"),
                doi=r.get("doi"),
                pmid=r.get("pmid"),
                pmcid=r.get("pmcid"),
                pub_year=str(r.get("pub_year")) if r.get("pub_year") else None,
                classification=r["classification"],
                classification_reason=r["classification_reason"],
                evidence=ev_obj,
                evidence_sentence=r.get("evidence_sentence"),
                evidence_status=r["evidence_status"],
                evidence_source=r.get("evidence_source", "abstract"),
                evidence_offsets=r.get("evidence_offsets"),
                text_sha256=r.get("text_sha256"),
                qualify_score=r.get("qualify_score"),
                confidence=conf_obj,
                confidence_band=r.get("confidence_band", "low"),
                confidence_kind=r.get("confidence_kind", "calibrated_platt_probability"),
                score_breakdown=sb_obj,
                guard_reason=r.get("guard_reason"),
                dedup_of=r.get("dedup_of"),
                k_eligible=r.get("k_eligible", 0),
                k_scored=r.get("k_scored", 0),
                reagent_signal=r.get("reagent_signal", 0.0),
            )
        )

    response = ClassifyResponse(
        entity=request.entity,
        query_entity=request.entity,
        criterion=request.criterion,
        intent=intent,
        query_used=query,
        entity_aliases_used=vetted_aliases,
        n_retrieved=len(raw_publications),
        total_retrieved=len(raw_publications),
        n_after_dedupe=len(publications),
        total_classified=len(pydantic_results),
        dedup_count=dedup_count,
        execution_time_ms=round(elapsed_ms, 2),
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
    Retrieves candidate publications, applies intent-aware extraction,
    and runs cross-encoder NLI inference offloaded to AnyIO threadpools.
    Accepts both entity and criterion in request body.
    """
    return await run_in_threadpool(_run_pipeline, request)


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    Atomically records user corrections to feedback.jsonl with thread safety.
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
