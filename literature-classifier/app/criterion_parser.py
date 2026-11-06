"""Scoped criterion parser and schema validator for G1 (product) and G2 (agent) grammars."""
import json
import os
import re
from typing import Optional, List, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field, model_validator


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SYSTEMS_PATH = os.path.join(CONFIG_DIR, "systems.json")


class SystemSpec(BaseModel):
    name: str
    family: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)


class CriterionSpec(BaseModel):
    entity: str
    entity_role: Literal["product", "agent"]
    agent_kind: Optional[Literal["catalyst", "reagent", "substrate"]] = None  # required iff role == "agent"
    activity: Optional[str] = None  # required iff role == "product"; optional for agent
    system: Optional[SystemSpec] = None  # None => not required
    scope: Literal["this_study", "any"] = "this_study"
    unparsed_clauses: List[str] = Field(default_factory=list)
    parse_method: Literal["rules", "user_supplied"] = "rules"
    parse_confidence: float = 1.0

    @model_validator(mode="after")
    def validate_role_fields(self) -> "CriterionSpec":
        if self.entity_role == "agent":
            if not self.agent_kind:
                raise ValueError("agent_kind ('catalyst', 'reagent', 'substrate') is required when entity_role == 'agent'")
        elif self.entity_role == "product":
            if not self.activity:
                raise ValueError("activity is required when entity_role == 'product'")
            if self.agent_kind is not None:
                raise ValueError("agent_kind must be None when entity_role == 'product'")
        return self


class CriterionParseError(ValueError):
    """Raised when criterion does not match supported grammar or contains unsupported clauses."""
    def __init__(self, message: str, unparsed_clauses: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.unparsed_clauses = unparsed_clauses or []


def load_systems_gazetteer() -> Dict[str, Any]:
    if os.path.exists(SYSTEMS_PATH):
        try:
            with open(SYSTEMS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"families": {}, "canonical_synonyms": {}}


SYSTEMS_DB = load_systems_gazetteer()


def match_system_in_text(system_str: str) -> SystemSpec:
    clean_sys = system_str.strip().lower()
    clean_sys = re.sub(r"^(engineered\s+|microbial\s+|recombinant\s+)", "", clean_sys)
    
    # Check exact family or organism
    matched_family = None
    for family, members in SYSTEMS_DB.get("families", {}).items():
        if clean_sys == family.lower() or any(clean_sys == m.lower() for m in members):
            matched_family = family
            break
            
    synonyms = SYSTEMS_DB.get("canonical_synonyms", {}).get(clean_sys, [])
    if not synonyms and matched_family:
        synonyms = [m for m in SYSTEMS_DB.get("families", {}).get(matched_family, []) if m != clean_sys]
        
    return SystemSpec(
        name=clean_sys,
        family=matched_family,
        synonyms=synonyms
    )


# Recognized unsupported clauses for testing and explicit error messages
UNSUPPORTED_PATTERNS = [
    r"\bwith\s+dose[- ]dependence\b",
    r"\bin\s+vitro\b",
    r"\bin\s+vivo\b",
    r"\bin\s+human\s+subjects\b",
    r"\bwith\s+enantiomeric\s+excess\s+>\s*\d+%\b",
    r"\bunder\s+continuous\s+flow\b",
    r"\bwith\s+pharmacokinetic\s+profiling\b"
]


def extract_unsupported_clauses(text: str) -> List[str]:
    found = []
    for pat in UNSUPPORTED_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            found.append(match.group(0).strip())
    return found


def parse_criterion_to_spec(
    criterion: str,
    entity: str,
    allow_partial_criterion: bool = False
) -> CriterionSpec:
    """
    Parse a natural language criterion into a structured CriterionSpec according to
    Enumerated Grammar G1 (product) and G2 (agent).
    
    G1 (product): <paper> reports <activity> of <entity> [in|using|by <system>] [in this study]
    G2 (agent): <paper> uses <entity> as a <catalyst|reagent|substrate> [in <activity>]
    """
    clean_crit = " ".join(criterion.strip().split())
    unparsed_clauses = extract_unsupported_clauses(clean_crit)
    
    if unparsed_clauses and not allow_partial_criterion:
        raise CriterionParseError(
            f"Unsupported criterion clause(s) detected: {', '.join(unparsed_clauses)}. "
            "Supported grammar shapes: G1 (product) and G2 (agent). "
            "Pass allow_partial_criterion=True to evaluate supported core.",
            unparsed_clauses=unparsed_clauses
        )

    # Detect scope
    scope: Literal["this_study", "any"] = "this_study"
    if re.search(r"\b(in\s+any\s+study|regardless\s+of\s+study|ever|in\s+literature)\b", clean_crit, re.IGNORECASE):
        scope = "any"

    # Check G2 (Agent role)
    # e.g. "The paper uses palladium(II) acetate as a catalyst in cross-coupling reactions"
    agent_match = re.search(
        r"(?:paper|study|authors|reports)?\s*(?:uses?|utilizes?|applied|employ(?:ed|s)?)\s+(.+?)\s+as\s+(?:a\s+|an\s+)?(catalyst|reagent|substrate)(?:\s+(?:in|for)\s+(.+?))?(?:\s+in\s+this\s+study)?\.?$",
        clean_crit,
        re.IGNORECASE
    )
    if agent_match:
        matched_entity = agent_match.group(1).strip()
        agent_kind = agent_match.group(2).lower()
        agent_activity = agent_match.group(3).strip() if agent_match.group(3) else None
        
        # Strip trailing "in this study" from activity if captured
        if agent_activity:
            agent_activity = re.sub(r"\s+in\s+this\s+study\.?$", "", agent_activity, flags=re.IGNORECASE).strip()

        return CriterionSpec(
            entity=entity or matched_entity,
            entity_role="agent",
            agent_kind=agent_kind,  # type: ignore
            activity=agent_activity,
            system=None,
            scope=scope,
            unparsed_clauses=unparsed_clauses,
            parse_method="rules",
            parse_confidence=1.0
        )

    # Check G1 (Product role)
    # e.g. "The paper describes the synthesis or production of vanillin."
    # e.g. "The paper reports microbial biosynthesis of artemisinic acid in yeast in this study."
    # e.g. "Psilocybin was synthesized or biosynthesized in this work."
    # e.g. "The compound was produced by engineered microbial biosynthesis in this study."
    g1_patterns = [
        # G1 full with optional system
        r"(?:The\s+)?(?:paper|study|article)?\s*(?:describes|reports|investigates|demonstrates)?\s*(?:the\s+)?(.+?)\s+of\s+(.+?)(?:\s+(?:in|by|using)\s+([a-zA-Z0-9\.\'\- ]+?))?(?:\s+in\s+this\s+(?:study|work))?\.?$",
        # Passive "was synthesized" or "production of X"
        r"(?:synthesis|production|biosynthesis|preparation|isolation|fermentation)\s+of\s+(.+?)(?:\s+(?:in|by|using)\s+([a-zA-Z0-9\.\'\- ]+?))?(?:\s+in\s+this\s+(?:study|work))?\.?$"
    ]

    for pat in g1_patterns:
        m = re.search(pat, clean_crit, re.IGNORECASE)
        if m:
            if len(m.groups()) == 3:
                activity_raw = m.group(1).strip()
                entity_raw = m.group(2).strip()
                system_raw = m.group(3).strip() if m.group(3) else None
            else:
                activity_raw = "synthesis or production"
                entity_raw = m.group(1).strip()
                system_raw = m.group(2).strip() if m.group(2) else None

            # Clean system_raw from trailing "in this study" or "in this work"
            system_spec = None
            if system_raw:
                sys_clean = re.sub(r"\s+in\s+this\s+(?:study|work)\.?$", "", system_raw, flags=re.IGNORECASE).strip()
                if sys_clean and sys_clean.lower() not in ("this study", "the study", "this work", "any"):
                    system_spec = match_system_in_text(sys_clean)

            # Ensure clean entity and activity
            final_activity = activity_raw
            if not final_activity or final_activity.lower() in ("the", "a", "an"):
                final_activity = "synthesis or production"

            return CriterionSpec(
                entity=entity or entity_raw,
                entity_role="product",
                agent_kind=None,
                activity=final_activity,
                system=system_spec,
                scope=scope,
                unparsed_clauses=unparsed_clauses,
                parse_method="rules",
                parse_confidence=1.0
            )

    # Fallback product match for standard assessment prompts
    if re.search(r"\b(synthesis|synthesized|production|produced|biosynthesis|biosynthesized|preparation|prepared|isolation|isolated|fermentation|fermented)\b", clean_crit, re.IGNORECASE):
        # Look for host system
        sys_spec = None
        sys_m = re.search(r"\b(?:in|using|by)\s+(yeast|bacteria|e\.\s*coli|s\.\s*cerevisiae|microbial|fungi|plant)\b", clean_crit, re.IGNORECASE)
        if sys_m:
            sys_spec = match_system_in_text(sys_m.group(1))

        # Extract activity keyword if available
        act_m = re.search(r"\b(chemical\s+synthesis|microbial\s+biosynthesis|enzymatic\s+biosynthesis|biosynthesis|synthesis|production|preparation|fermentation)\b", clean_crit, re.IGNORECASE)
        activity_str = act_m.group(1) if act_m else "synthesis or production"

        return CriterionSpec(
            entity=entity,
            entity_role="product",
            agent_kind=None,
            activity=activity_str,
            system=sys_spec,
            scope=scope,
            unparsed_clauses=unparsed_clauses,
            parse_method="rules",
            parse_confidence=0.85
        )

    # If it completely fails to match any supported shape:
    raise CriterionParseError(
        f"Criterion '{clean_crit}' does not match supported grammar shapes: "
        "G1: '<paper> reports <activity> of <entity> [in <system>] [in this study]' or "
        "G2: '<paper> uses <entity> as a <catalyst|reagent|substrate> [in <activity>]'.",
        unparsed_clauses=unparsed_clauses
    )

