"""Hypothesis construction, criterion intent detection, and role pattern matching."""
import re
import os
import json
import difflib
import logging
from typing import List, Tuple, Optional, Dict, Any, Set
from .text_utils import vet_aliases

logger = logging.getLogger(__name__)

# Synthesis action stems (NOTE: 'obtain' is deliberately omitted from stems to prevent spurious arming)
SYNTH_STEMS = (
    "synthes", "produc", "ferment", "biosynth", "express", "extract", "yield",
    "isolat", "biotransform", "bioconver", "generat", "manufactur", "engineer",
    "prepar", "purif", "fabricat", "catalyz", "cultivat"
)

# Reference dictionary for restricted fuzzy matching (tokens >= 6 chars only)
SYNTH_WORDS = [
    "synthesis", "synthesize", "synthesizing", "synthesized",
    "production", "produce", "producing", "produced",
    "fermentation", "fermenting", "biosynthesis", "biosynthesizing",
    "extraction", "extracting", "isolation", "isolating",
    "manufacture", "manufacturing", "manufactured",
    "purification", "purifying", "biotransformation", "bioconversion"
]

MIN_FUZZY_LEN = 6

DEFAULT_SYNTH_PARAPHRASES = [
    "{entity} was synthesized in this study.",
    "{entity} was prepared as the product of the reaction.",
    "{entity} was produced or biosynthesized in this work."
]

DEFAULT_NEGATIVE_ROLE_HYPOTHESIS = "{entity} is used as a reagent, catalyst, or starting material."

DEFAULT_CONTRAST_TEMPLATES = [
    "{entity} was used as a starting material to synthesize other compounds.",
    "{entity} is used as a starting material to produce other products.",
    "{entity} is used as a precursor, reagent, or catalyst in this study.",
    "{entity} is administered or added to the system, not generated."
]

DEFAULT_ROLE_PATTERNS = [
    r"in the presence of\s+(?:the\s+)?{E}",
    r"catalyz(?:ed|ed by)\s+(?:by\s+)?{E}",
    r"mediated by\s+(?:the\s+)?{E}",
    r"using\s+(?:the\s+)?{E}\s+as\s+(?:a\s+)?(?:catalyst|reagent|precursor|substrate|starting material)",
    r"using\s+{E}\b(?!\s+(?:was|is)\s+(?:synthesized|produced|prepared))",
    r"treated with\s+(?:the\s+)?{E}",
    r"(?:immobili[sz]ation|impregnation) of\s+{E}",
    r"with\s+{E}\s+as\s+(?:a\s+)?(?:catalyst|reagent|precursor|substrate)",
    r"{E}[- ]catalyzed",
    r"{E}\s+serves as\s+(?:a\s+)?(?:starting material|precursor|reagent|catalyst|substrate)",
    r"{E}\s+was (?:used|employed)\s+as\s+(?:a\s+)?(?:starting material|precursor|reagent|catalyst|substrate)",
    r"used\s+{E}\s+as\s+(?:a\s+)?(?:starting material|precursor|reagent|catalyst|substrate)",
    r"addition of\s+(?:the\s+)?{E}",
    r"supplemented with\s+(?:the\s+)?{E}",
    r"fed (?:with\s+)?(?:diets containing\\s+)?{E}"
]

_templates_cache: Optional[Dict[str, Any]] = None
_intent_map_cache: Optional[Dict[str, Any]] = None
_role_patterns_cache: Optional[List[str]] = None


def load_templates_config() -> Dict[str, Any]:
    """Load templates from config/templates.json if present, else fallback to defaults."""
    global _templates_cache
    if _templates_cache is not None:
        return _templates_cache

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "templates.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                _templates_cache = json.load(f)
                return _templates_cache
        except Exception as e:
            logger.warning(f"Failed to read templates.json, using defaults: {e}")
    _templates_cache = {
        "synthesis_paraphrases": DEFAULT_SYNTH_PARAPHRASES,
        "negative_role_hypothesis": DEFAULT_NEGATIVE_ROLE_HYPOTHESIS,
        "default_contrast_hypotheses": DEFAULT_CONTRAST_TEMPLATES
    }
    return _templates_cache


def load_intent_map() -> Dict[str, Any]:
    """Load intent mapping configuration from config/intent_map.json."""
    global _intent_map_cache
    if _intent_map_cache is not None:
        return _intent_map_cache

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "intent_map.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                _intent_map_cache = json.load(f)
                return _intent_map_cache
        except Exception as e:
            logger.warning(f"Failed to read intent_map.json, using defaults: {e}")
    _intent_map_cache = {
        "synthesis": {
            "intent_triggers": ["synthesis", "synthesize", "production", "produce", "fermentation", "biosynthesis"],
            "search_terms": ["synthesis", "biosynthesis", "\"produced by\"", "fermentation", "preparation", "yielded"],
            "requires_production_verb": True
        },
        "pharmacology": {
            "intent_triggers": ["activity", "inhibition", "treatment", "efficacy", "pharmacology", "mice"],
            "search_terms": ["activity", "treatment", "inhibition", "efficacy", "pharmacology"],
            "requires_production_verb": False
        }
    }
    return _intent_map_cache


def load_role_patterns() -> List[str]:
    """Load regex role patterns from config/role_patterns.json."""
    global _role_patterns_cache
    if _role_patterns_cache is not None:
        return _role_patterns_cache

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "role_patterns.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                _role_patterns_cache = json.load(f)
                return _role_patterns_cache
        except Exception as e:
            logger.warning(f"Failed to read role_patterns.json, using defaults: {e}")
    _role_patterns_cache = DEFAULT_ROLE_PATTERNS
    return _role_patterns_cache


def detect_intent(criterion: str, entity: str = "") -> str:
    """
    Classify user criterion intent: 'synthesis', 'pharmacology', or 'other'.
    """
    if not criterion:
        return "other"
    intent_map = load_intent_map()

    crit_lower = criterion.lower()
    # Check synthesis triggers first
    synth_cfg = intent_map.get("synthesis", {})
    synth_triggers = synth_cfg.get("intent_triggers", [])
    if any(re.search(rf"\b{re.escape(t)}\b", crit_lower) for t in synth_triggers) or lexical_intent_armed(criterion, entity):
        return "synthesis"

    # Check pharmacology triggers
    pharm_cfg = intent_map.get("pharmacology", {})
    pharm_triggers = pharm_cfg.get("intent_triggers", [])
    if any(re.search(rf"\b{re.escape(t)}\b", crit_lower) for t in pharm_triggers):
        return "pharmacology"

    return "other"


def lexical_intent_armed(criterion: str, entity: str = "") -> bool:
    """
    Determine if a criterion describes synthesis/production via stems and restricted fuzzy matching.
    """
    if not criterion:
        return False
    toks = re.findall(r"[a-z]+", criterion.lower())
    for t in toks:
        if any(t.startswith(stem) for stem in SYNTH_STEMS):
            return True
        if len(t) >= MIN_FUZZY_LEN:
            matches = difflib.get_close_matches(t, SYNTH_WORDS, n=1, cutoff=0.85)
            if matches:
                return True

    # 'obtain' is only accepted if it immediately co-occurs with the entity in the criterion
    if "obtain" in criterion.lower() and entity:
        ent_toks = set(re.findall(r"[a-z]+", entity.lower()))
        crit_toks = set(toks)
        if ent_toks.intersection(crit_toks):
            words = criterion.lower().split()
            for idx, w in enumerate(words):
                if "obtain" in w:
                    window = " ".join(words[max(0, idx - 4): min(len(words), idx + 5)])
                    if entity.lower() in window:
                        return True

    return False


def check_role_patterns(sentence: str, entity: str, aliases: Optional[List[str]] = None) -> Optional[str]:
    """
    Check if the sentence matches any negative-role patterns for the entity (e.g. catalyst, reagent, substrate).
    Returns the matching pattern or descriptive reason string, or None if no match.
    """
    patterns = load_role_patterns()
    vetted_aliases = vet_aliases(aliases, entity)
    terms = [entity] + vetted_aliases

    # Build escaped regex group for target entity and aliases
    escaped_terms = [re.escape(t.strip()) for t in terms if t.strip()]
    if not escaped_terms:
        return None
    entity_regex = r"(?:" + "|".join(escaped_terms) + r")"

    for pattern_template in patterns:
        try:
            pat_str = pattern_template.format(E=entity_regex)
            match = re.search(pat_str, sentence, re.IGNORECASE)
            if match:
                return f"role_pattern: {match.group(0)}"
        except Exception as e:
            logger.debug(f"Pattern matching error on {pattern_template}: {e}")
            continue

    return None


def get_negative_role_hypothesis(entity: str) -> str:
    """Get the negative role hypothesis for comparison."""
    cfg = load_templates_config()
    hyp = cfg.get("negative_role_hypothesis", DEFAULT_NEGATIVE_ROLE_HYPOTHESIS)
    return hyp.format(entity=entity).strip()


def build_hypotheses(
    criterion: str,
    entity: str,
    synthesis_armed: bool,
    user_paraphrases: Optional[List[str]] = None
) -> List[str]:
    """
    Build the intent-gated hypothesis ensemble.
    Preserves full context and avoids dropping qualifiers.
    """
    hyps: List[str] = [criterion.strip()]

    if synthesis_armed:
        cfg = load_templates_config()
        templates = cfg.get("synthesis_paraphrases", DEFAULT_SYNTH_PARAPHRASES)
        for t in templates:
            formatted = t.format(entity=entity).strip()
            if formatted not in hyps:
                hyps.append(formatted)

    if user_paraphrases:
        for p in user_paraphrases:
            clean_p = p.strip()
            if clean_p and clean_p not in hyps:
                hyps.append(clean_p)

    return hyps


def load_contrast_hypotheses(
    entity: str,
    custom_hypotheses: Optional[List[str]] = None
) -> List[str]:
    """Load affirmative contrast hypotheses formatted for the target entity."""
    if custom_hypotheses:
        return [h.format(entity=entity).strip() for h in custom_hypotheses if h.strip()]

    cfg = load_templates_config()
    templates = cfg.get("default_contrast_hypotheses", DEFAULT_CONTRAST_TEMPLATES)
    return [t.format(entity=entity).strip() for t in templates]
