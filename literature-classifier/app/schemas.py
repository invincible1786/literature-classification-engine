"""Pydantic v2 schemas for request validation, structured responses, and feedback logging."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def sync_max_k(self) -> "ClassifyRequest":
        if self.max_k is not None:
            self.max_results = self.max_k
        return self


class EvidenceSpan(BaseModel):
    """Verbatim extracted textual evidence span."""
    text: str = Field(..., description="Exact verbatim substring from document text.")
    char_start: int = Field(..., description="0-based start character offset in canonical document text.")
    char_end: int = Field(..., description="0-based end character offset in canonical document text.")
    source: str = Field(default="abstract", description="'abstract', 'fulltext', or other document section.")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ConfidenceDetails(BaseModel):
    """Calibrated probability score and monotone qualitative band."""
    score: float = Field(..., description="Calibrated P(correct) in [0.0, 1.0].")
    band: str = Field(..., description="'high' (>=0.80), 'medium' (0.50-0.80), or 'low' (<0.50).")

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
    """Structured classification result for an individual publication."""
    paper_id: str
    rank: int = Field(default=1, description="1-based deterministic result rank.")
    title: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    pub_year: Optional[str] = None
    classification: str = Field(..., description="'qualifies' or 'does_not_qualify'")
    classification_reason: str = Field(
        ...,
        description="Diagnostic explanation (e.g. 'margin_above_threshold', 'disqualified_by_contrast_guard', 'insufficient_evidence')."
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
        default="abstract",
        description="'abstract', 'fulltext', or other document section."
    )
    evidence_offsets: Optional[List[int]] = Field(
        default=None,
        description="Exact [start, end] character slice offsets in canonical document text."
    )
    text_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of canonicalized document text for verifiable auditing."
    )
    qualify_score: Optional[float] = Field(
        default=None,
        description="Monotonic ranking score in [0.0, 1.0], or null if insufficient text."
    )
    confidence: Optional[Any] = Field(
        default=None,
        description="Confidence details object {score, band} or numeric float for backward compatibility."
    )
    confidence_band: str = Field(
        default="low",
        description="'high', 'medium', 'low', or 'uncalibrated'."
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

    @model_validator(mode="after")
    def sync_compatibility_fields(self) -> "PaperClassificationResult":
        # Mirror evidence object to evidence_sentence and evidence_offsets
        if self.evidence is not None:
            if not self.evidence_sentence:
                self.evidence_sentence = self.evidence.text
            if not self.evidence_offsets:
                self.evidence_offsets = [self.evidence.char_start, self.evidence.char_end]
            if not self.evidence_source:
                self.evidence_source = self.evidence.source
        elif self.evidence_sentence and not self.evidence:
            start = self.evidence_offsets[0] if self.evidence_offsets and len(self.evidence_offsets) == 2 else 0
            end = self.evidence_offsets[1] if self.evidence_offsets and len(self.evidence_offsets) == 2 else len(self.evidence_sentence)
            self.evidence = EvidenceSpan(text=self.evidence_sentence, char_start=start, char_end=end, source=self.evidence_source or "abstract")

        # Mirror confidence object
        if isinstance(self.confidence, dict):
            self.confidence_band = self.confidence.get("band", self.confidence_band)
        elif isinstance(self.confidence, ConfidenceDetails):
            self.confidence_band = self.confidence.band
        elif isinstance(self.confidence, (int, float)):
            score_val = float(self.confidence)
            band_val = "high" if score_val >= 0.80 else ("medium" if score_val >= 0.50 else "low")
            self.confidence = ConfidenceDetails(score=score_val, band=band_val)
            self.confidence_band = band_val

        return self


class ClassifyResponse(BaseModel):
    """Top-level API response envelope."""
    entity: str = Field(default="", description="Target entity name.")
    query_entity: str = Field(default="", description="Target entity name (backwards compatibility).")
    criterion: str = Field(..., description="Classification criterion.")
    intent: str = Field(default="synthesis", description="Detected intent ('synthesis', 'pharmacology', 'other').")
    query_used: str = Field(default="", description="Actual Lucene query used for retrieval.")
    model_revision: str = Field(..., description="Pinned model revision hash.")
    calibration_version: str = Field(default="platt-v1-seed30", description="Calibration model version.")
    n_retrieved: int = Field(default=0, description="Total raw papers retrieved from Europe PMC.")
    total_retrieved: int = Field(default=0, description="Total raw papers retrieved (backwards compatibility).")
    n_after_dedupe: int = Field(default=0, description="Unique papers remaining after deduplication.")
    total_classified: int = Field(default=0, description="Total papers classified (backwards compatibility).")
    dedup_count: int = Field(default=0, description="Number of duplicate preprints/articles absorbed.")
    entity_aliases_used: List[str] = Field(default_factory=list, description="Vetted entity aliases used in query and scanning.")
    execution_time_ms: float = Field(..., description="Total pipeline execution latency in milliseconds.")
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

        return self


class FeedbackRequest(BaseModel):
    """User correction submission for active learning and audit logging."""
    paper_id: str = Field(..., min_length=1, max_length=100)
    entity: str = Field(..., min_length=1, max_length=100)
    criterion: str = Field(..., min_length=3, max_length=500)
    suggested_label: str = Field(..., pattern="^(qualifies|does_not_qualify)$")
    notes: Optional[str] = Field(default=None, max_length=1000)
