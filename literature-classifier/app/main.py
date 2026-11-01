import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from app.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    FeedbackRequest,
    FeedbackResponse,
    PaperClassificationResult
)
from app.retrieval import build_query, search_europe_pmc, RetrievalError
from app.text_utils import init_punkt
from app.nli import DebertaNLIScorer, NLIScorer, DEFAULT_MODEL_NAME, DEFAULT_REVISION
from app.classifier import (
    classify_publications,
    DEFAULT_THRESHOLD,
    DEFAULT_CONTRAST_DELTA,
    EvidenceIntegrityError,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("literature-classifier")

# Global scorer instance (or via app.state)
scorer: Optional[NLIScorer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan warmup:
    Initializes NLTK punkt sentence tokenizer and loads the cross-encoder NLI model
    onto CPU/CUDA before serving requests.
    """
    global scorer
    logger.info("Initializing NLP components and warm-up...")
    init_punkt()
    scorer = DebertaNLIScorer()
    app.state.scorer = scorer
    logger.info("Lifespan warmup complete. Ready to serve traffic.")
    yield
    logger.info("Shutting down literature classification engine.")


app = FastAPI(
    title="Mini Literature Classification Engine",
    description="Semantically classifies scientific literature against user criteria using Europe PMC and Cross-Encoder NLI.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Service health check and model metadata."""
    active_scorer = getattr(app.state, "scorer", scorer)
    device = getattr(active_scorer, "device", "cpu") if active_scorer else "uninitialized"
    return HealthResponse(
        status="healthy",
        model_name=DEFAULT_MODEL_NAME,
        model_revision=DEFAULT_REVISION,
        device=str(device)
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """
    Retrieve publications from Europe PMC for the entity and classify each
    against the inclusion criterion.
    """
    query = build_query(request.entity, request.synonyms)
    logger.info(f"Classify request: entity='{request.entity}', query='{query}', max_results={request.max_results}")

    try:
        publications = search_europe_pmc(
            query=query,
            max_results=request.max_results,
            include_fulltext=request.include_fulltext
        )
    except RetrievalError as err:
        logger.error(f"Retrieval error from Europe PMC: {err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Europe PMC service error: {err}"
        )

    # Handle zero results gracefully with HTTP 200
    if not publications:
        return ClassifyResponse(
            query_used=query,
            normalised_criterion=request.criterion,
            threshold=request.threshold if request.threshold is not None else DEFAULT_THRESHOLD,
            contrast_delta=request.contrast_delta if request.contrast_delta is not None else DEFAULT_CONTRAST_DELTA,
            hypotheses_used=[],
            contrast_guard_active="active",
            total_retrieved=0,
            note=f"No publications found for query {query}",
            results=[]
        )

    active_scorer = getattr(app.state, "scorer", scorer)
    if active_scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLI model is not yet initialized."
        )

    try:
        run_output = classify_publications(
            publications=publications,
            entity=request.entity,
            criterion=request.criterion,
            synonyms=request.synonyms,
            threshold=request.threshold,
            contrast_delta=request.contrast_delta,
            contrast_template=request.contrast_hypothesis,
            contrast_hypotheses=request.contrast_hypotheses,
            scorer=active_scorer
        )
    except EvidenceIntegrityError as err:
        logger.critical(f"Evidence integrity check failed: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence verification failed: {err}"
        )

    ranked_results = run_output.results
    norm_crit = run_output.normalised_criterion
    tau = run_output.tau
    delta = run_output.delta
    hypotheses_used = run_output.hypotheses_used
    contrast_guard_active = run_output.contrast_guard_active

    clean_results = [
        PaperClassificationResult(
            title=res["title"],
            pmid=res["pmid"],
            doi=res["doi"],
            pmcid=res["pmcid"],
            classification=res["classification"],
            evidence=res["evidence"],
            evidence_status=res["evidence_status"],
            evidence_source=res.get("evidence_source", "abstract"),
            qualify_score=res["qualify_score"],
            confidence=res["confidence"],
            confidence_band=res["confidence_band"],
        )
        for res in ranked_results
    ]

    return ClassifyResponse(
        query_used=query,
        normalised_criterion=norm_crit,
        threshold=tau,
        contrast_delta=delta,
        hypotheses_used=hypotheses_used,
        contrast_guard_active=contrast_guard_active,
        total_retrieved=len(clean_results),
        note=None,
        results=clean_results
    )


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackRequest):
    """
    Collects user feedback on a classification prediction to store in JSONL.
    Enables future evaluation set expansion, threshold re-tuning, and calibration.
    """
    feedback_file = os.path.join(os.path.dirname(__file__), "..", "feedback.jsonl")
    record = feedback.model_dump()
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return FeedbackResponse(
            status="success",
            message=f"Feedback recorded for paper {feedback.paper_id}"
        )
    except Exception as exc:
        logger.error(f"Failed to record feedback: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not persist feedback: {exc}"
        )
