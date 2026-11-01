import re
from typing import List


def normalise_criterion(criterion: str, entity: str) -> str:
    """
    Light normalization of user criterion:
    - If it contains the literal '{entity}', substitute it.
    - Never auto-inject entity if '{entity}' is not present.
    - Strip leading conversational framing like 'The paper reports that', 'This study shows that', etc.
    """
    c = criterion.strip()
    if "{entity}" in c:
        c = c.replace("{entity}", entity.strip())

    framing_regex = r'^(?:the paper reports that|this study reports that|this paper reports that|the authors report that|this study shows that|this paper shows that|in this work,?)\s+'
    c = re.sub(framing_regex, '', c, flags=re.IGNORECASE).strip()

    if c and c[0].islower():
        c = c[0].upper() + c[1:]
    return c


def generate_criterion_paraphrases(criterion: str, entity: str) -> List[str]:
    """
    Generate canonical deterministic paraphrases of the criterion.
    Ensemble scoring across these forms stabilizes zero-shot cross-encoder NLI,
    collapsing phrasing-sensitivity spread.
    """
    base = normalise_criterion(criterion, entity)
    base_clean = base.rstrip(".")

    return [
        base,
        f"In this study, the authors carried out the work described as: {base_clean}.",
        f"This paper reports as its own experimental result: {base_clean}."
    ]


SYNTHESIS_KEYWORD_REGEX = re.compile(
    r'(?:synthes|produc|biosynthes|isolat|obtain|prepar|ferment|yield)',
    re.IGNORECASE
)


def is_synthesis_criterion(criterion: str) -> bool:
    """Check if the inclusion criterion relates to chemical or biological synthesis."""
    return bool(SYNTHESIS_KEYWORD_REGEX.search(criterion))
