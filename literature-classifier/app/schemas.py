"""Pydantic v2 schemas for request validation, structured responses, and feedback logging."""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator

from app.criterion_parser import CriterionSpec
from app.reasons import validate_reason_code

CALIBRATION_VERSION_CANONICAL = "platt-v3.2-mle"
NEGATIVE_CALIBRATED_REASONS = ("different_activity", "wrong_system")


class ClassifyRequest(BaseModel):
    """Input payload for literature classification."""
    entity: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target chemical or biological entity name (e.g. 'vanillin', 'chitin', 'caffeine')."
    )
    criterion: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language inclusion criterion (e.g. 'The paper describes the synthesis or production of vanillin.')."
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Maximum number of candidate papers to retrieve and classify (capped at 25)."
    )
    max_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=25,
        description="Alias for max_results."
    )
    aliases: Optional[List[str]] = Field(
        default=None,
        description="Optional list of manual chemical or biological entity synonyms."
    )
    expand_aliases: bool = Field(
        default=True,
        description="Whether to query PubChem PUG REST for automated synonym expansion."
    )
    include_fulltext: bool = Field(
        default=False,
        description="Whether to attempt open-access PMC XML full-text retrieval for papers with PMCID."
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Decision threshold tau. If omitted, uses default calibrated value (0.45)."
    )
    contrast_delta: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Affirmative contrast guard significance delta. If omitted, uses 0.00."
    )
    custom_contrast_hypotheses: Optional[List[str]] = Field(
        default=None,
        description="Optional user-defined contrast hypotheses to override default reagent templates."
    )
    user_paraphrases: Optional[List[str]] = Field(
        default=None,
        description="Optional additional criterion paraphrases to include in the ensemble."
    )
    offline: bool = Field(
        default=False,
        description="If True, loads pre-indexed offline snapshot from data/snapshots/ without network requests."
    )
    pipeline: Literal["v2", "v3"] = Field(
        default="v3",
        description="Pipeline version to run: 'v3' (default multi-condition 3-class) or 'v2' (legacy binary)."
    )
    abstract_negative_policy: Literal["strict", "conservative"] = Field(
        default="strict",
        description="D6 policy for abstract-only host mismatches: 'strict' (does_not_qualify/wrong_system, default) or 'conservative' (system_unstated)."
    )
    allow_partial_criterion: bool = Field(
        default=False,
        description="If True, allows evaluating supported core when unparsed clauses exist (caps qualifies to insufficient_evidence)."
    )
    criterion_spec: Optional[CriterionSpec] = Field(
        default=None,
        description="Optional pre-parsed or user-supplied CriterionSpec overriding rule-based parser."
    )

    @model_validator(mode="after")
    def sync_max_k(self) -> "ClassifyRequest":
        if self.max_k is not None:
            self.max_results = self.max_k
        return self


class EvidenceSpan(BaseModel):
    """Verbatim extracted textual evidence span supporting v2 and v3 schemas."""
    text: str = Field(default="", description="Exact verbatim substring from document text (display_text).")
    char_start: int = Field(default=0, description="0-based start character offset in canonical document text.")
    char_end: int = Field(default=0, description="0-based end character offset in canonical document text.")
    source: str = Field(default="abstract", description="'abstract', 'fulltext', or other document section.")
    
    # v3 fields
    role: Optional[str] = Field(default="activity", description="'entity', 'activity', or 'system'.")
    raw_text: Optional[str] = Field(default=None, description="Verbatim raw text including markup if present.")
    display_text: Optional[str] = Field(default=None, description="Cleaned verbatim text with markup stripped.")
    raw_start: Optional[int] = Field(default=None, description="Start offset in uncleaned raw source record.")
    raw_end: Optional[int] = Field(default=None, description="End offset in uncleaned raw source record.")
    sentence_index: int = Field(default=0, description="0-based sentence index in section.")
    verifiable_alone: bool = Field(default=True, description="Whether the span entails its condition independently.")

    @model_validator(mode="after")
    def sync_span_fields(self) -> "EvidenceSpan":
        if not self.display_text and self.text:
            self.display_text = self.text
        elif not self.text and self.display_text:
            self.text = self.display_text
            
        if self.raw_start is None:
            self.raw_start = self.char_start
        if self.raw_end is None:
            self.raw_end = self.char_end
            
        if not self.raw_text:
            self.raw_text = self.display_text or self.text
        return self

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ConfidenceDetails(BaseModel):
    """Calibrated probability score and monotone qualitative band."""
    score: Optional[float] = Field(default=None, description="Calibrated P(correct) in [0.0, 1.0], or null.")
    band: str = Field(default="Low", description="'High' (>=0.80), 'Medium' (0.50-0.80), or 'Low' (<0.50).")
    value: Optional[float] = Field(default=None, description="Numeric confidence value in [0.0, 1.0], or null.")
    method: str = Field(default="platt_scaling", description="Calibration methodology ('platt_scaling', 'not_fitted').")
    calibration_version: str = Field(default=CALIBRATION_VERSION_CANONICAL, description="Calibration model revision.")

    @model_validator(mode="after")
    def sync_confidence_values(self) -> "ConfidenceDetails":
        if self.value is None and self.score is not None:
            self.value = self.score
        elif self.score is None and self.value is not None:
            self.score = self.value
        return self

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ScoreBreakdown(BaseModel):
    """Granular decomposition of sentence and gate scoring."""
    positive_margin: Optional[float] = Field(default=None, description="P(entail) - P(contradict) against positive paraphrases.")
    role_score: Optional[float] = Field(default=None, description="Entailment score on negative-role hypothesis.")
    final_score: Optional[float] = Field(default=None, description="Sentence margin after negative role and gate penalties.")
    gate_flags: List[str] = Field(default_factory=list, description="Diagnostic flags such as 'this_work_positive' or 'review_abstain'.")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class PaperClassificationResult(BaseModel):
    """Structured classification result for an individual publication in v2 and v3 format."""
    api_version: str = Field(default="v3", description="API format version ('v3').")
    paper_id: str
    rank: int = Field(default=1, description="1-based deterministic result rank.")
    title: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    pub_year: Optional[str] = None
    classification: str = Field(..., description="'qualifies', 'does_not_qualify', or 'insufficient_evidence'")
    classification_reason: str = Field(
        default="",
        description="Diagnostic explanation."
    )
    reason_code: Optional[str] = Field(
        default=None,
        description="Registered machine-readable reason code from app/reasons.py."
    )
    criterion_spec: Optional[CriterionSpec] = Field(
        default=None,
        description="Parsed CriterionSpec driving the decision."
    )
    evidence: Optional[EvidenceSpan] = Field(
        default=None,
        description="Exact verbatim evidence object, or null when insufficient/no_text."
    )
    evidence_sentence: Optional[str] = Field(
        default=None,
        description="Exact verbatim sentence supporting the decision (backwards compatibility mirror of evidence.text)."
    )
    evidence_status: str = Field(
        ...,
        description="'supported', 'contradicted', 'insufficient', or 'no_text'."
    )
    evidence_source: str = Field(
        default="abstract_only",
        description="'abstract_only' or 'fulltext'."
    )
    evidence_offsets: Optional[List[int]] = Field(
        default=None,
        description="Exact [start, end] character slice offsets."
    )
    source_record_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of uncleaned raw source record."
    )
    text_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of canonicalized document text."
    )
    aliases_used: List[str] = Field(
        default_factory=list,
        description="Vetted aliases used for matching."
    )
    retrieved_by: str = Field(
        default="intent",
        description="'intent', 'plain', or 'both'."
    )
    conditions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Condition decomposition {entity, activity, system}."
    )
    qualify_score: Optional[float] = Field(
        default=None,
        description="Monotonic ranking score in [0.0, 1.0], or null if insufficient text."
    )
    confidence: Optional[Any] = Field(
        default=None,
        description="Confidence details object {value, band, method, calibration_version} or null."
    )
    confidence_band: str = Field(
        default="Low",
        description="'High', 'Medium', 'Low', or 'uncalibrated'."
    )
    confidence_details: Optional[ConfidenceDetails] = Field(
        default=None,
        description="Granular confidence details object {score, band}."
    )
    confidence_kind: str = Field(
        default="calibrated_platt_probability",
        description="Honest statement of uncertainty quantification methodology."
    )
    score_breakdown: Optional[ScoreBreakdown] = Field(
        default=None,
        description="Detailed score breakdown (positive_margin, role_score, final_score, gate_flags)."
    )
    guard_reason: Optional[str] = Field(
        default=None,
        description="Which role pattern or guard triggered disqualification."
    )
    diagnostics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagnostic information including resolved antecedents and structural guard firings."
    )
    dedup_of: Optional[str] = Field(
        default=None,
        description="Identifier of duplicate publication absorbed by this record."
    )
    k_eligible: int = Field(
        default=0,
        description="Total number of eligible entity-bearing sentences discovered in the document."
    )
    k_scored: int = Field(
        default=0,
        description="Number of candidate sentences scored after intent-aware selection."
    )
    reagent_signal: float = Field(
        default=0.0,
        description="Peak contrast entailment probability c* on the best candidate sentence."
    )
    timings_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage latency profiling."
    )

    @model_validator(mode="after")
    def sync_compatibility_fields(self) -> "PaperClassificationResult":
        # Ensure classification_reason populated if empty
        if not self.classification_reason:
            if self.reason_code:
                self.classification_reason = self.reason_code
            else:
                self.classification_reason = self.classification

        # Mirror evidence object to evidence_sentence and evidence_offsets
        # F1: When evidence_status == "insufficient" or "no_text", set top-level evidence fields to null
        if self.evidence_status in ("insufficient", "no_text"):
            self.evidence = None
            self.evidence_sentence = None
            self.evidence_offsets = None
        elif self.evidence is not None:
            if not self.evidence_sentence:
                self.evidence_sentence = self.evidence.text
            if not self.evidence_offsets:
                self.evidence_offsets = [self.evidence.char_start, self.evidence.char_end]
            if not self.evidence_source:
                self.evidence_source = self.evidence.source
        elif self.evidence_sentence and not self.evidence:
            if not str(self.evidence_sentence).startswith("No "):
                start = self.evidence_offsets[0] if self.evidence_offsets and len(self.evidence_offsets) == 2 else 0
                end = self.evidence_offsets[1] if self.evidence_offsets and len(self.evidence_offsets) == 2 else len(self.evidence_sentence)
                self.evidence = EvidenceSpan(text=self.evidence_sentence, char_start=start, char_end=end, source=self.evidence_source or "abstract")

        # Mirror confidence fields
        if isinstance(self.confidence, dict):
            band_str = self.confidence.get("band", self.confidence_band)
            self.confidence_band = band_str.capitalize() if band_str else "Low"
            score_val = self.confidence.get("score") if self.confidence.get("score") is not None else self.confidence.get("value")
            if score_val is not None:
                self.confidence_details = ConfidenceDetails(
                    score=float(score_val),
                    value=float(score_val),
                    band=str(band_str).capitalize(),
                    method=self.confidence.get("method", "platt_scaling"),
                    calibration_version=self.confidence.get("calibration_version", CALIBRATION_VERSION_CANONICAL)
                )
        elif isinstance(self.confidence, ConfidenceDetails):
            self.confidence_band = self.confidence.band.capitalize()
            self.confidence_details = self.confidence
        elif isinstance(self.confidence, (int, float)):
            score_val = float(self.confidence)
            band_val = "High" if score_val >= 0.80 else ("Medium" if score_val >= 0.50 else "Low")
            if not self.confidence_band or self.confidence_band == "low":
                self.confidence_band = band_val
            if not self.confidence_details:
                self.confidence_details = ConfidenceDetails(score=score_val, value=score_val, band=self.confidence_band)

        # Enforce D5 confidence table and reason registry invariants in v3+
        if self.api_version == "v3":
            if self.classification == "qualifies" and self.reason_code is not None:
                raise ValueError("reason_code must be None when classification == 'qualifies'")
            if self.classification != "qualifies" and self.reason_code is not None:
                if not validate_reason_code(self.reason_code):
                    raise ValueError(f"Unregistered reason_code: {self.reason_code}")

            # Extract numeric confidence if present
            has_numeric_conf = False
            if isinstance(self.confidence, (int, float)):
                has_numeric_conf = True
            elif isinstance(self.confidence, dict) and (
                self.confidence.get("score") is not None or self.confidence.get("value") is not None
            ):
                has_numeric_conf = True
            elif isinstance(self.confidence, ConfidenceDetails) and (
                self.confidence.score is not None or self.confidence.value is not None
            ):
                has_numeric_conf = True

            if self.classification == "insufficient_evidence" and has_numeric_conf:
                raise ValueError("D5 invariant: insufficient_evidence must emit confidence=None")
            if (
                self.classification == "does_not_qualify"
                and self.reason_code is not None
                and self.reason_code not in NEGATIVE_CALIBRATED_REASONS
                and has_numeric_conf
            ):
                raise ValueError(
                    f"D5 invariant: categorical/veto negative '{self.reason_code}' must emit confidence=None"
                )

        return self


class ClassifyResponse(BaseModel):
    """Top-level API response envelope conforming to v3 with v2 backwards compatibility."""
    api_version: str = Field(default="v3", description="API format version ('v3').")
    entity: str = Field(default="", description="Target entity name.")
    query_entity: str = Field(default="", description="Target entity name (backwards compatibility).")
    criterion: str = Field(..., description="Classification criterion.")
    criterion_spec: Optional[CriterionSpec] = Field(default=None, description="Structured CriterionSpec.")
    intent: str = Field(default="synthesis", description="Detected intent ('synthesis', 'pharmacology', 'other').")
    query_used: str = Field(default="", description="Actual Lucene query used for retrieval.")
    query_plain: Optional[str] = Field(default=None, description="Plain entity Lucene query used in dual retrieval.")
    model_revision: str = Field(..., description="Pinned model revision hash.")
    calibration_version: str = Field(default=CALIBRATION_VERSION_CANONICAL, description="Calibration model version.")
    n_retrieved: int = Field(default=0, description="Total raw papers retrieved from Europe PMC.")
    total_retrieved: int = Field(default=0, description="Total raw papers retrieved (backwards compatibility).")
    n_after_dedupe: int = Field(default=0, description="Unique papers remaining after deduplication.")
    total_classified: int = Field(default=0, description="Total papers classified (backwards compatibility).")
    dedup_count: int = Field(default=0, description="Number of duplicate preprints/articles absorbed.")
    entity_aliases_used: List[str] = Field(default_factory=list, description="Vetted entity aliases used in query and scanning.")
    aliases_used: List[str] = Field(default_factory=list, description="Vetted entity aliases (v3 schema).")
    retrieved_by: str = Field(default="both", description="Dual retrieval mode ('intent', 'plain', or 'both').")
    execution_time_ms: float = Field(..., description="Total pipeline execution latency in milliseconds.")
    timings_ms: Dict[str, float] = Field(default_factory=dict, description="Granular per-stage latencies in ms.")
    ensemble_size: int = Field(default=1, description="Number of criterion hypotheses evaluated in ensemble.")
    hypotheses_used: List[str] = Field(default_factory=list, description="List of positive paraphrase hypotheses.")
    contrast_guard_active: bool = Field(default=True, description="Whether the contrast guard / role filter was active.")
    guard_arming: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic guard arming status.")
    results: List[PaperClassificationResult] = Field(..., description="Ranked classification results.")

    @model_validator(mode="after")
    def sync_response_fields(self) -> "ClassifyResponse":
        if not self.entity and self.query_entity:
            self.entity = self.query_entity
        elif not self.query_entity and self.entity:
            self.query_entity = self.entity

        if not self.n_retrieved and self.total_retrieved:
            self.n_retrieved = self.total_retrieved
        elif not self.total_retrieved and self.n_retrieved:
            self.total_retrieved = self.n_retrieved

        if not self.n_after_dedupe and self.total_classified:
            self.n_after_dedupe = self.total_classified
        elif not self.total_classified and self.n_after_dedupe:
            self.total_classified = self.n_after_dedupe

        if not self.aliases_used and self.entity_aliases_used:
            self.aliases_used = self.entity_aliases_used
        elif not self.entity_aliases_used and self.aliases_used:
            self.entity_aliases_used = self.aliases_used

        return self


class FeedbackRequest(BaseModel):
    """User correction submission for active learning and curator review queue."""
    paper_id: str = Field(..., min_length=1, max_length=100)
    entity: str = Field(..., min_length=1, max_length=100)
    criterion: str = Field(..., min_length=3, max_length=500)
    suggested_label: str = Field(..., pattern="^(qualifies|does_not_qualify|insufficient_evidence)$")
    notes: Optional[str] = Field(default=None, max_length=1000)
