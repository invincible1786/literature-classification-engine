import re
import html
import logging
from typing import List, Tuple, Optional
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

logger = logging.getLogger(__name__)

# Common abbreviations in biomedical / scientific literature to prevent spurious splits
SCIENTIFIC_ABBREVIATIONS = {
    'al', 'fig', 'figs', 'e.g', 'i.e', 'ca', 'vs', 'ref', 'refs',
    'dr', 'prof', 'no', 'vol', 'suppl', 'et', 'sp', 'spp', 'subsp',
    'mol', 'wt', 'approx', 'min', 'sec', 'hr', 'hrs', 'conc'
}

_punkt_tokenizer: Optional[PunktSentenceTokenizer] = None


def init_punkt() -> PunktSentenceTokenizer:
    """Initialize and warm up NLTK Punkt tokenizer with scientific abbreviations."""
    global _punkt_tokenizer
    if _punkt_tokenizer is None:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        params = PunktParameters()
        params.abbrev_types = SCIENTIFIC_ABBREVIATIONS
        _punkt_tokenizer = PunktSentenceTokenizer(params)
    return _punkt_tokenizer


def clean_text(raw_text: Optional[str]) -> str:
    """
    Strip HTML tags, unescape HTML entities, normalize whitespace, and trim.
    Returns empty string for None or whitespace-only input.
    """
    if not raw_text:
        return ""
    # Unescape HTML entities (&amp;, &lt;, &gt;, etc.)
    text = html.unescape(raw_text)
    # Strip HTML tags like <i>, <b>, <p>, etc.
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse multiple whitespaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_entity_regex(entity: str, synonyms: Optional[List[str]] = None) -> re.Pattern:
    """
    Build a case-insensitive regex pattern with word boundaries for the entity
    and any user-provided synonyms.
    Escapes regex metacharacters, supports hyphen/space flexibility,
    and allows optional plural endings while preventing false token substring matches.
    """
    terms: List[str] = [entity.strip()]
    if synonyms:
        for s in synonyms:
            clean_s = s.strip()
            if clean_s and clean_s.lower() != entity.strip().lower() and clean_s not in terms:
                terms.append(clean_s)

    subpatterns: List[str] = []
    for term in terms:
        if not term:
            continue
        # Escape special regex characters first
        escaped = re.escape(term)
        # Allow hyphens and spaces to match flexibly: e.g. "4-phosphoryloxy" or "4 phosphoryloxy"
        escaped = re.sub(r'(\\\s+|\\-)', r'[- ]?', escaped)
        # Optional plural if term ends with alphanumeric
        plural = r'(?:s)?' if term[-1].isalnum() else ''
        # Use lookaround boundaries to safely handle terms with punctuation e.g. (+)-psilocybin
        subpatterns.append(rf'(?:(?<!\w){escaped}{plural}(?!\w))')

    pattern_str = '|'.join(subpatterns)
    return re.compile(pattern_str, re.IGNORECASE)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using the warmed-up scientific Punkt tokenizer."""
    if not text:
        return []
    tokenizer = init_punkt()
    raw_sentences = tokenizer.tokenize(text)
    cleaned = []
    for s in raw_sentences:
        s_clean = s.strip()
        # Discard trivially short or non-alphanumeric fragments
        if len(s_clean) > 3 and any(c.isalnum() for c in s_clean):
            cleaned.append(s_clean)
    return cleaned


def candidate_sentences(
    title: Optional[str],
    abstract: Optional[str],
    entity_regex: re.Pattern,
    max_candidates: int = 10
) -> Tuple[List[str], bool, int]:
    """
    Extract candidate sentences for NLI classification.
    Returns:
      (candidates, title_has_entity, n_entity_sentences)

    Rules:
    - If no abstract and no entity in title, or entity only matched via metadata -> candidates is []
    - Includes title (if entity present in title, or title + entity-bearing abstract sentences)
    - Pre-ranks abstract sentences by entity density and position, capped at max_candidates
    """
    clean_title = clean_text(title)
    clean_abs = clean_text(abstract)

    title_has_entity = bool(entity_regex.search(clean_title)) if clean_title else False
    abs_sentences = split_sentences(clean_abs) if clean_abs else []

    # Identify abstract sentences containing the entity
    entity_sentences_with_meta: List[Tuple[str, int, float]] = []
    for idx, sent in enumerate(abs_sentences):
        matches = list(entity_regex.finditer(sent))
        if matches:
            # entity density = match count / sentence length
            density = len(matches) / max(len(sent), 1)
            entity_sentences_with_meta.append((sent, idx, density))

    n_entity_sentences = len(entity_sentences_with_meta) + (1 if title_has_entity else 0)

    # If neither title has entity nor abstract has entity sentences -> insufficient text / metadata only
    if not title_has_entity and not entity_sentences_with_meta:
        return ([], False, 0)

    # Rank entity sentences: by density descending, then earlier position ascending
    # Sentence at beginning of abstract usually has higher informative value
    sorted_entity_sentences = sorted(
        entity_sentences_with_meta,
        key=lambda item: (-item[2], item[1])
    )
    ranked_abs_sentences = [item[0] for item in sorted_entity_sentences]

    # Combine title and abstract sentences
    candidates: List[str] = []
    if clean_title:
        candidates.append(clean_title)

    for sent in ranked_abs_sentences:
        if sent not in candidates:
            candidates.append(sent)
        if len(candidates) >= max_candidates:
            break

    # Cap at max_candidates
    return (candidates[:max_candidates], title_has_entity, n_entity_sentences)

