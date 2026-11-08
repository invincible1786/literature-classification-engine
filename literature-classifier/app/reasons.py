"""Reason code registry for v3 classification outcomes (veto, evidence-based negative, abstain)."""
from dataclasses import dataclass
from typing import Dict, Literal, Optional, List


@dataclass(frozen=True)
class ReasonCodeInfo:
    code: str
    label: Literal["does_not_qualify", "insufficient_evidence"]
    kind: Literal["veto", "evidence-based", "abstain", "metadata"]
    description: str
    fixture_example: str


# Canonical registry of all valid v3 reason codes
REASON_REGISTRY: Dict[str, ReasonCodeInfo] = {
    "no_text": ReasonCodeInfo(
        code="no_text",
        label="insufficient_evidence",
        kind="abstain",
        description="The publication has no usable abstract or full text.",
        fixture_example="Paper with null or empty title and abstract."
    ),
    "abstract_only": ReasonCodeInfo(
        code="abstract_only",
        label="insufficient_evidence",
        kind="abstain",
        description="Abstract does not provide decisive evidence and full-text is unavailable.",
        fixture_example="Abstract notes study of metabolic behavior without naming specific chemical yields."
    ),
    "mention_only": ReasonCodeInfo(
        code="mention_only",
        label="insufficient_evidence",
        kind="abstain",
        description="The entity is mentioned topically without any experimental activity claim.",
        fixture_example="Network pharmacology paper discussing target receptors across disease states."
    ),
    "activity_unclear": ReasonCodeInfo(
        code="activity_unclear",
        label="insufficient_evidence",
        kind="abstain",
        description="Activity claims in the text are ambiguous, hedged, or insufficiently entailed.",
        fixture_example="May possibly contribute to the synthesis under hypothetical conditions."
    ),
    "entity_unclear": ReasonCodeInfo(
        code="entity_unclear",
        label="insufficient_evidence",
        kind="abstain",
        description="Target entity was not matched in the abstract, but full text is unavailable (entity may be in Methods).",
        fixture_example="Abstract refers generally to phenolic volatiles without explicitly naming vanillin."
    ),
    "entity_absent": ReasonCodeInfo(
        code="entity_absent",
        label="does_not_qualify",
        kind="evidence-based",
        description="Target entity is confirmed absent from the complete full text of the publication.",
        fixture_example="Full text verified to contain zero mentions of the target entity or vetted aliases."
    ),
    "system_unstated": ReasonCodeInfo(
        code="system_unstated",
        label="insufficient_evidence",
        kind="abstain",
        description="The criterion requires a specific biological/chemical host system, but the abstract does not state one.",
        fixture_example="Paper synthesizes vanillin, but does not state in abstract whether yeast or bacteria was used."
    ),
    "system_mismatch_abstract": ReasonCodeInfo(
        code="system_mismatch_abstract",
        label="insufficient_evidence",
        kind="abstain",
        description="Abstract mentions an alternative host organism (e.g. E. coli when yeast required), but under conservative policy full-text may have used yeast.",
        fixture_example="Abstract reports production in E. coli while criterion specified Saccharomyces cerevisiae."
    ),
    "conflicting_evidence": ReasonCodeInfo(
        code="conflicting_evidence",
        label="insufficient_evidence",
        kind="abstain",
        description="Document contains direct self-contradictions or dueling evidence on the same condition.",
        fixture_example="One sentence reports production of X, while another states X was not detected under tested conditions."
    ),
    "evidence_split": ReasonCodeInfo(
        code="evidence_split",
        label="insufficient_evidence",
        kind="abstain",
        description="Activity sentence refers to the entity only anaphorically (e.g. 'the product') and no single sentence binds entity to activity.",
        fixture_example="'The product was obtained in 80% yield' following a sentence naming multiple candidate precursors."
    ),
    "unsupported_criterion_clause": ReasonCodeInfo(
        code="unsupported_criterion_clause",
        label="insufficient_evidence",
        kind="abstain",
        description="Under allow_partial_criterion, the paper meets the supported core but has unparsed clauses preventing qualification.",
        fixture_example="Criterion includes 'with dose-dependence' which cannot be evaluated; caps result to insufficient_evidence."
    ),
    "different_activity": ReasonCodeInfo(
        code="different_activity",
        label="does_not_qualify",
        kind="evidence-based",
        description="Verbatim this-work evidence proves a different activity (e.g. degradation, pharmacology, clinical trial) while requested activity has zero support.",
        fixture_example="'In this clinical trial, caffeine administration was evaluated for pharmacokinetic tolerance in human volunteers.'"
    ),
    "wrong_system": ReasonCodeInfo(
        code="wrong_system",
        label="does_not_qualify",
        kind="evidence-based",
        description="Full text (or strict policy) confirms an alternative host/system was used, and requested system is not supported.",
        fixture_example="Full text proves production was carried out exclusively in mammalian cell lines, not microbial host."
    ),
    "reagent_role": ReasonCodeInfo(
        code="reagent_role",
        label="does_not_qualify",
        kind="veto",
        description="Rule veto: target entity appears as the grammatical object of an explicit prepositional reagent/catalyst pattern.",
        fixture_example="'In the presence of palladium(II) acetate, substituted dibenzoxepines were prepared in 78% yield.'"
    ),
    "background_citation": ReasonCodeInfo(
        code="background_citation",
        label="insufficient_evidence",
        kind="abstain",
        description="Activity mention is an epistemic citation of prior art ('previously reported', 'has been produced') rather than primary experimental finding of this study.",
        fixture_example="'Artemisinic acid has been produced by engineered yeast in previous studies (Ro et al., 2006).'"
    ),
    "possible_review": ReasonCodeInfo(
        code="possible_review",
        label="insufficient_evidence",
        kind="abstain",
        description="High review lexical score combined with low experimental attribution cues demotes activity to unknown.",
        fixture_example="'We review recent advances in microbial terpene pathways and their commercial prospects.'"
    ),
    "review_article": ReasonCodeInfo(
        code="review_article",
        label="does_not_qualify",
        kind="veto",
        description="Rule veto: publication metadata pubType=review with no first-person experimental cues in scope of this_study.",
        fixture_example="Europe PMC pubType='Review' discussing synthetic literature without primary lab experimentation."
    ),
}


def get_reason_info(code: Optional[str]) -> Optional[ReasonCodeInfo]:
    """Retrieve reason metadata from registry; returns None if code is None or unregistered."""
    if not code:
        return None
    return REASON_REGISTRY.get(code)


def validate_reason_code(code: str) -> bool:
    """Check if code is an officially registered v3 reason code."""
    return code in REASON_REGISTRY


def list_registered_codes() -> List[str]:
    """List all registered reason codes."""
    return sorted(list(REASON_REGISTRY.keys()))

