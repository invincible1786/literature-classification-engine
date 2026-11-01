from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    entity: str = Field(..., description="Target chemical entity or term, e.g. 'psilocybin'")
    criterion: str = Field(..., description="Inclusion criterion, e.g. 'Psilocybin is synthesised or biosynthesised in this work'")
    synonyms: List[str] = Field(default_factory=list, description="Synonyms or alternative spellings for the entity")
    max_results: int = Field(default=15, ge=1, le=50, description="Max publications to retrieve from Europe PMC (1 to 50)")
    threshold: Optional[float] = Field(default=None, description="Custom classification decision margin threshold tau (defaults to tuned value)")
    contrast_delta: Optional[float] = Field(default=None, description="Guard margin delta for contrast hypothesis (defaults to tuned value)")
    contrast_hypothesis: Optional[str] = Field(
        default=None,
        description="Legacy single contrast hypothesis template. Substituted with {entity}."
    )
    contrast_hypotheses: Optional[List[str]] = Field(
        default=None,
        description="List of custom contrast hypothesis templates. If None, uses default multi-hypothesis ensemble."
    )
    include_fulltext: bool = Field(
        default=False,
        description="Whether to retrieve open-access full-text XML for papers with PMCID."
    )

    @field_validator("entity", "criterion")
    @classmethod
    def check_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Field must not be empty or whitespace only")
        return s


class PaperClassificationResult(BaseModel):
    title: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    classification: Literal["qualifies", "does_not_qualify"]
    evidence: Optional[str] = None
    evidence_status: Literal["entailing_sentence", "contrast_mention", "strongest_mention", "insufficient_text"]
    evidence_source: Literal["abstract", "fulltext"] = "abstract"
    qualify_score: Optional[float] = Field(None, description="Ranking score in [0, 1] computed as (m + 1) / 2, or None for insufficient_text")
    confidence: Optional[float] = Field(None, description="Symmetric confidence in [0.5, 1.0], or None for insufficient_text")
    confidence_band: Literal["high", "medium", "low", "insufficient"]


class ClassifyResponse(BaseModel):
    query_used: str
    normalised_criterion: str
    threshold: float
    contrast_delta: float
    hypotheses_used: List[str] = Field(default_factory=list, description="Paraphrases and hypotheses evaluated")
    contrast_guard_active: str = Field(default="active", description="'active', 'custom', or 'not_applicable'")
    total_retrieved: int
    note: Optional[str] = None
    results: List[PaperClassificationResult]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_revision: str
    device: str


class FeedbackRequest(BaseModel):
    paper_id: str = Field(..., description="PMID, DOI, or PMCID identifying the publication")
    criterion: str = Field(..., description="Criterion used during classification")
    predicted: Literal["qualifies", "does_not_qualify"]
    user_label: Literal["qualifies", "does_not_qualify"]
    corrected_evidence: Optional[str] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str

