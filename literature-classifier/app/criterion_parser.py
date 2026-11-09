"""Scoped criterion parser and schema validator for G1 (product), G2 (agent), and G3 (modifiers)."""
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


class ClauseConstraint(BaseModel):
    """G3 modifier constraint extracted from user criterion (WS14-D)."""
    kind: Literal["exclusion", "numeric_threshold", "scope"]
    target: str
    operator: Optional[Literal[">", ">=", "<", "<="]] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    raw: str


class CriterionSpec(BaseModel):
    entity: str
    entity_role: Literal["product", "agent"]
    agent_kind: Optional[Literal["catalyst", "reagent", "substrate"]] = None  # required iff role == "agent"
    activity: Optional[str] = None  # required iff role == "product"; optional for agent
    system: Optional[SystemSpec] = None  # None => not required
    method: Optional[str] = None  # e.g. "chemical" (F11)
    scope: Literal["this_study", "any"] = "this_study"
    constraints: List[ClauseConstraint] = Field(default_factory=list)
    unparsed_clauses: List[str] = Field(default_factory=list)
    parse_method: Literal["rules", "user_supplied"] = "rules"
    parse_confidence: float = 1.0

    @property
    def system_family(self) -> Optional[str]:
        return self.system.family if self.system else None

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
    if clean_sys in ("vitro", "in-vitro"):
        clean_sys = "in vitro"
    clean_sys_lookup = clean_sys

    db = load_systems_gazetteer()
    families = db.get("families", {})
    matched_family = None
    # 1. Exact family name match takes precedence (e.g., 'yeast' -> 'yeast', not umbrella 'microbial')
    for family in families:
        if clean_sys_lookup == family.lower():
            matched_family = family
            break
    # 2. Fallback to member match across families
    if matched_family is None:
        for family, members in families.items():
            if any(clean_sys_lookup == m.lower() for m in members):
                matched_family = family
                break

    synonyms = list(db.get("canonical_synonyms", {}).get(clean_sys_lookup, []))
    if matched_family:
        for m in families.get(matched_family, []):
            if m.lower() != clean_sys_lookup and m not in synonyms:
                synonyms.append(m)

    return SystemSpec(
        name=clean_sys,
        family=matched_family,
        synonyms=synonyms
    )


# Recognized unsupported clauses (Note: 'in vitro' is supported via WS14-B in_vitro family)
UNSUPPORTED_PATTERNS = [
    r"\bwith\s+dose[- ]dependence\b",
    r"\bin\s+vivo\b",
    r"\bin\s+human\s+subjects\b",
    r"\bwith\s+enantiomeric\s+excess\s*[><=]+\s*\d+%",
    r"\bunder\s+continuous\s+flow\b",
    r"\bwith\s+pharmacokinetic\s+profiling\b",
]


def extract_unsupported_clauses(text: str) -> List[str]:
    found = []
    for pat in UNSUPPORTED_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            found.append(match.group(0).strip())
    return found


def normalize_numeric_value(val: float, raw_unit: str) -> Tuple[float, str, bool]:
    """
    Normalize recognized scientific quantity units to canonical units.
    Returns: (normalized_value, canonical_unit, is_supported)
    """
    u = raw_unit.strip()
    u_low = u.lower().replace(" ", "")
    if u_low in ("g/l", "g/liter"):
        return val, "g/L", True
    if u_low in ("mg/l", "mg/liter"):
        return val / 1000.0, "g/L", True
    if u_low in ("ug/l", "μg/l", "mcg/l"):
        return val / 1_000_000.0, "g/L", True
    if u_low == "%":
        return val, "%", True
    if u_low in ("mol%", "mol.%"):
        return val, "mol%", True
    return val, u, False


_OP_MAP: Dict[str, Literal[">", ">=", "<", "<="]] = {
    ">": ">",
    ">=": ">=",
    "≥": ">=",
    "<": "<",
    "<=": "<=",
    "≤": "<=",
    "above": ">",
    "at least": ">=",
    "below": "<",
    "at most": "<=",
}

_RE_NUMERIC_CLAUSE = re.compile(
    r"(?:(?:and|;|,)\s*)?(?:with\s+)?\b(yield|titer|titre|conversion|selectivity|purity|concentration)\s+"
    r"(>=|<=|≥|≤|>|<|at\s+least|at\s+most|above|below)\s*"
    r"(\d+(?:\.\d+)?)\s*(g/L|mg/L|ug/L|μg/L|mol%|%|mM|uM|kg/h)(?=\s|$|[,;.])",
    re.IGNORECASE,
)

_RE_EXCLUSION_CLAUSE = re.compile(
    r"(?:(?:and|;|,)\s*)?\b(?:excluding|but\s+not(?:\s+via)?|not\s+via)\s+"
    r"([a-zA-Z0-9\-\s]+?)(?=\s*(?:,|;|\band\b|\bwith\b|\bin\s+this\s+(?:study|work)\b|$|\.))",
    re.IGNORECASE,
)


def extract_g3_constraints(text: str) -> Tuple[str, List[ClauseConstraint], List[str]]:
    """
    Parse G3 modifier constraints (exclusions and numeric thresholds) and strip them
    from the criterion string so G1/G2 core grammar matches cleanly.
    Returns: (stripped_core_text, constraints, unparsed_unit_clauses)
    """
    constraints: List[ClauseConstraint] = []
    unparsed: List[str] = []
    working = text

    # 1. Extract exclusions
    for m in list(_RE_EXCLUSION_CLAUSE.finditer(text)):
        target = " ".join(m.group(1).strip().lower().split())
        if target:
            constraints.append(ClauseConstraint(
                kind="exclusion",
                target=target,
                raw=m.group(0).strip(" ,;")
            ))
            working = working.replace(m.group(0), " ")

    # 2. Extract numeric thresholds
    for m in list(_RE_NUMERIC_CLAUSE.finditer(working)):
        target = m.group(1).strip().lower()
        if target == "titre":
            target = "titer"
        op_raw = " ".join(m.group(2).strip().lower().split())
        op = _OP_MAP.get(op_raw, ">")
        val = float(m.group(3))
        unit_raw = m.group(4).strip()
        norm_val, norm_unit, supported = normalize_numeric_value(val, unit_raw)
        if supported:
            constraints.append(ClauseConstraint(
                kind="numeric_threshold",
                target=target,
                operator=op,
                value=norm_val,
                unit=norm_unit,
                raw=m.group(0).strip(" ,;")
            ))
        else:
            unparsed.append(m.group(0).strip(" ,;"))
        working = working.replace(m.group(0), " ")

    # Clean up leftover punctuation/connectors after stripping modifiers
    working = re.sub(r"\s*,\s*(?=(in\s+this\s+(?:study|work)|\.|$))", " ", working, flags=re.IGNORECASE)
    working = re.sub(r"\b(?:and|;)\s*(?=(in\s+this\s+(?:study|work)|\.|$))", " ", working, flags=re.IGNORECASE)
    working = re.sub(r"\s+", " ", working).strip()
    return working, constraints, unparsed


def parse_criterion_to_spec(
    criterion: str,
    entity: str,
    allow_partial_criterion: bool = False
) -> CriterionSpec:
    """
    Parse a natural language criterion into a structured CriterionSpec according to
    Enumerated Grammar G1 (product), G2 (agent), and G3 (modifiers).
    """
    clean_crit = " ".join(criterion.strip().split())
    unparsed_clauses = extract_unsupported_clauses(clean_crit)

    # Extract G3 modifier constraints before core G1/G2 matching
    core_crit, constraints, g3_unparsed = extract_g3_constraints(clean_crit)
    for uc in g3_unparsed:
        if uc not in unparsed_clauses:
            unparsed_clauses.append(uc)

    if unparsed_clauses and not allow_partial_criterion:
        raise CriterionParseError(
            f"Unsupported criterion clause(s) detected: {', '.join(unparsed_clauses)}. "
            "Supported grammar shapes: G1 (product), G2 (agent), and G3 (modifiers). "
            "Pass allow_partial_criterion=True to evaluate supported core.",
            unparsed_clauses=unparsed_clauses
        )

    # Detect scope
    scope: Literal["this_study", "any"] = "this_study"
    if re.search(r"\b(in\s+any\s+study|regardless\s+of\s+study|ever|in\s+literature)\b", core_crit, re.IGNORECASE):
        scope = "any"

    # F10: Pre-nominal system adjectives bind spec.system
    prenom_match = re.search(
        r"\b(engineered\s+microbial|microbial|yeast|bacterial|plant|cell-free|cell\s+free|chemoenzymatic|enzymatic)\s+"
        r"(?:bio)?(?:synthesis|production|fermentation)\b",
        clean_crit,
        re.IGNORECASE
    )
    prenom_sys = None
    if prenom_match:
        mod = prenom_match.group(1).lower().strip()
        if mod in ("microbial", "engineered microbial", "yeast", "bacterial"):
            prenom_sys = match_system_in_text(mod)
            if not prenom_sys.family:
                prenom_sys = SystemSpec(name=mod, family="microbial", synonyms=[])
        elif mod == "plant":
            prenom_sys = match_system_in_text("plant")
        elif mod in ("cell-free", "cell free", "enzymatic", "chemoenzymatic"):
            prenom_sys = match_system_in_text("in vitro")

    # F11: Route requirement (chemical route auto-excludes biological pathways)
    chem_match = re.search(
        r"\b(?:chemical(?:ly)?\s+(?:synthesized|prepared|produced)|chemical\s+synthesis)\b",
        core_crit,
        re.IGNORECASE
    )
    method_str = None
    if chem_match:
        method_str = "chemical"
        auto_excls = ["biosynthesis", "fermentation", "enzymatic", "plant extraction", "microbial", "in vivo"]
        existing_targets = {c.target.lower() for c in constraints if c.kind == "exclusion"}
        for excl in auto_excls:
            if excl not in existing_targets:
                constraints.append(ClauseConstraint(
                    kind="exclusion",
                    target=excl,
                    raw=f"chemical route excludes {excl}"
                ))

    # Check G2 (Agent role)
    agent_match = re.search(
        r"(?:paper|study|authors|reports)?\s*(?:uses?|utilizes?|applied|employ(?:ed|s)?)\s+(.+?)\s+as\s+(?:a\s+|an\s+)?(catalyst|reagent|substrate)(?:\s+(?:in|for)\s+(.+?))?(?:\s+in\s+this\s+(?:study|work))?\.?$",
        core_crit,
        re.IGNORECASE
    )
    if agent_match:
        matched_entity = agent_match.group(1).strip()
        agent_kind = agent_match.group(2).lower()
        agent_activity = agent_match.group(3).strip() if agent_match.group(3) else None

        if agent_activity:
            agent_activity = re.sub(r"\s+in\s+this\s+(?:study|work)\.?$", "", agent_activity, flags=re.IGNORECASE).strip()

        return CriterionSpec(
            entity=entity or matched_entity,
            entity_role="agent",
            agent_kind=agent_kind,  # type: ignore
            activity=agent_activity,
            system=None,
            scope=scope,
            constraints=constraints,
            unparsed_clauses=unparsed_clauses,
            parse_method="rules",
            parse_confidence=1.0
        )

    # Check G1 (Product role)
    g1_patterns = [
        r"(?:The\s+)?(?:paper|study|article)?\s*(?:describes|reports|investigates|demonstrates)?\s*(?:the\s+)?(.+?)\s+of\s+(.+?)(?:\s+(?:in|by|using)\s+([a-zA-Z0-9\.\'\- ]+?))?(?:\s+in\s+this\s+(?:study|work))?\.?$",
        r"(?:synthesis|production|biosynthesis|preparation|isolation|fermentation)\s+of\s+(.+?)(?:\s+(?:in|by|using)\s+([a-zA-Z0-9\.\'\- ]+?))?(?:\s+in\s+this\s+(?:study|work))?\.?$"
    ]

    for pat in g1_patterns:
        m = re.search(pat, core_crit, re.IGNORECASE)
        if m:
            if len(m.groups()) == 3:
                activity_raw = m.group(1).strip()
                entity_raw = m.group(2).strip()
                system_raw = m.group(3).strip() if m.group(3) else None
            else:
                activity_raw = "synthesis or production"
                entity_raw = m.group(1).strip()
                system_raw = m.group(2).strip() if m.group(2) else None

            system_spec = None
            if system_raw:
                sys_clean = re.sub(r"\s+in\s+this\s+(?:study|work)\.?$", "", system_raw, flags=re.IGNORECASE).strip()
                if sys_clean and sys_clean.lower() not in ("this study", "the study", "this work", "any"):
                    system_spec = match_system_in_text(sys_clean)

            final_activity = activity_raw
            if not final_activity or final_activity.lower() in ("the", "a", "an"):
                final_activity = "synthesis or production"

            if system_spec is None and prenom_sys is not None:
                system_spec = prenom_sys

            return CriterionSpec(
                entity=entity or entity_raw,
                entity_role="product",
                agent_kind=None,
                activity=final_activity,
                system=system_spec,
                method=method_str,
                scope=scope,
                constraints=constraints,
                unparsed_clauses=unparsed_clauses,
                parse_method="rules",
                parse_confidence=1.0
            )

    # Fallback product match for passive / standard assessment prompts
    if re.search(r"\b(synthesis|synthesized|production|produced|biosynthesis|biosynthesized|preparation|prepared|isolation|isolated|fermentation|fermented)\b", core_crit, re.IGNORECASE):
        sys_spec = None
        sys_m = re.search(
            r"\b(?:in|using|by)\s+(yeast|bacteria|e\.\s*coli|s\.\s*cerevisiae|microbial|fungi|plant|in\s+vitro|cell-free|cell\s+free)\b",
            core_crit,
            re.IGNORECASE
        )
        if sys_m:
            sys_spec = match_system_in_text(sys_m.group(1))

        if sys_spec is None and prenom_sys is not None:
            sys_spec = prenom_sys

        act_m = re.search(
            r"\b(chemical\s+synthesis|microbial\s+biosynthesis|enzymatic\s+biosynthesis|biosynthesis|synthesis|production|preparation|fermentation)\b",
            core_crit,
            re.IGNORECASE
        )
        activity_str = act_m.group(1) if act_m else "synthesis or production"

        return CriterionSpec(
            entity=entity,
            entity_role="product",
            agent_kind=None,
            activity=activity_str,
            system=sys_spec,
            method=method_str,
            scope=scope,
            constraints=constraints,
            unparsed_clauses=unparsed_clauses,
            parse_method="rules",
            parse_confidence=0.85
        )

    raise CriterionParseError(
        f"Criterion '{clean_crit}' does not match supported grammar shapes: "
        "G1: '<paper> reports <activity> of <entity> [in <system>] [in this study]' or "
        "G2: '<paper> uses <entity> as a <catalyst|reagent|substrate> [in <activity>]'.",
        unparsed_clauses=unparsed_clauses
    )


parse_criterion = parse_criterion_to_spec
