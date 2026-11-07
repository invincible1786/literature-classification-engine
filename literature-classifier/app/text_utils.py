"""Text utilities: canonicalization, domain sentence segmentation, alias vetting, and entity matching."""
import re
import os
import html
import unicodedata
import logging
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict, Any
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

logger = logging.getLogger(__name__)

# Inline tags removed with NO gap (e.g. <i>E. coli</i> -> E. coli; 10<sup>-3</sup> -> 10-3)
INLINE_TAGS = re.compile(r"</?(?:i|b|em|strong|sup|sub)\b[^>]*>", re.I)
# Other tags replaced with a single space to avoid merging adjacent words
ANY_TAG = re.compile(r"<[^>]+>")

# Biomedical and scientific abbreviation set
SCIENTIFIC_ABBREVIATIONS = {
    "aq", "equiv", "cat", "approx", "ca", "sp", "spp", "subsp", "var", "cv",
    "ref", "refs", "resp", "ie", "eg", "dr", "mr", "ms", "inc", "ltd", "co",
    "std", "dev", "temp", "conc", "wt", "vol", "mol", "min", "sec", "no",
    "fig", "figs", "eq", "eqs", "cf", "vs", "al"
}

_punkt_tokenizer: Optional[PunktSentenceTokenizer] = None
_alias_denylist: Optional[Set[str]] = None


@dataclass(frozen=True)
class Sent:
    """A sentence span bound to canonical document text."""
    text: str
    start: int
    end: int
    sentence_index: int = 0
    source: str = "abstract"


@dataclass(frozen=True)
class TextSegment:
    """Section-aware text segment with raw and canonical boundary mapping."""
    source: str  # 'abstract', 'intro', 'methods', 'results', 'discussion', 'caption', 'title'
    raw_start: int
    raw_end: int
    canon_start: int
    canon_end: int


def load_alias_denylist() -> Set[str]:
    """Load denylisted alias words and collisions from config/alias_denylist.txt."""
    global _alias_denylist
    if _alias_denylist is not None:
        return _alias_denylist

    denylist = {
        "thein", "theine", "marinus thein", "mat", "aid", "lead", "acid", "salt",
        "base", "yield", "water", "cell", "light", "dark", "medium", "extract",
        "tea", "coffee", "standard", "control", "blank", "sample", "plant",
        "animal", "human", "first person"
    }

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "alias_denylist.txt")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip().lower()
                    if line and not line.startswith("#"):
                        denylist.add(line)
        except Exception as e:
            logger.warning(f"Could not load alias_denylist.txt: {e}")

    _alias_denylist = denylist
    return _alias_denylist


def vet_aliases(raw_aliases: Optional[List[str]], entity: str) -> List[str]:
    """
    Vet candidate aliases:
    - Minimum length >= 5 characters
    - Must not be in alias denylist
    - Must not equal target entity (case-insensitive)
    - Must contain letters and not be purely numbers or punctuation
    """
    if not raw_aliases:
        return []

    denylist = load_alias_denylist()
    entity_clean = entity.strip().lower()
    vetted: List[str] = []

    for alias in raw_aliases:
        if not alias:
            continue
        clean_alias = alias.strip()
        lower_alias = clean_alias.lower()

        # Reject exact match with entity
        if lower_alias == entity_clean:
            continue

        # Reject short aliases (< 5 chars)
        if len(clean_alias) < 5:
            continue

        # Reject denylisted words and collisions
        if lower_alias in denylist:
            continue

        # Reject single common word surnames/names if in denylist
        if any(term in denylist for term in lower_alias.split()):
            continue

        # Must have at least some alphabetic characters
        if not any(c.isalpha() for c in clean_alias):
            continue

        if clean_alias not in vetted:
            vetted.append(clean_alias)

    return vetted


def clean_markup_and_unescape(raw: Optional[str]) -> str:
    """Clean markup and unescape HTML before canonicalization/indexing."""
    if not raw:
        return ""
    t = INLINE_TAGS.sub("", raw)
    t = ANY_TAG.sub(" ", t)
    t = html.unescape(t)
    return unicodedata.normalize("NFKC", t)


def canonicalize(raw: Optional[str]) -> str:
    """
    Canonicalize raw text into a clean, normalized string.
    Single-pass rule:
    1. Strip inline tags (i, b, sup, sub) with NO gap.
    2. Convert all other HTML/XML tags into a single space.
    3. Unescape HTML entities exactly ONCE.
    4. Normalize Unicode to NFKC.
    5. Collapse multiple whitespaces and strip.
    """
    if not raw:
        return ""
    t = clean_markup_and_unescape(raw)
    return re.sub(r"\s+", " ", t).strip()


def compute_sha256(text: str) -> str:
    """Compute standard SHA-256 hex digest of raw string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def map_canonical_span_to_raw(raw_text: str, canon_start: int, canon_end: int) -> Tuple[int, int]:
    """
    Map character slice offsets from canonical document text back to enclosing
    character boundaries in the uncleaned raw document text.
    Snaps outward to include enclosing tags/formatting if necessary.
    """
    if not raw_text:
        return 0, 0

    canon_doc = canonicalize(raw_text)
    if canon_start >= len(canon_doc) or canon_end > len(canon_doc) or canon_start >= canon_end:
        return 0, len(raw_text)

    target_sent = canon_doc[canon_start:canon_end].strip()
    if not target_sent:
        return 0, 0

    # 1. Exact verbatim match in raw_text
    idx = raw_text.find(target_sent)
    if idx != -1:
        return idx, idx + len(target_sent)

    # 2. Case-insensitive exact match in raw_text
    m_exact = re.search(re.escape(target_sent), raw_text, re.IGNORECASE)
    if m_exact:
        return m_exact.start(), m_exact.end()

    # 3. Match with whitespace normalization (handles XML linebreaks/multiple spaces)
    words = target_sent.split()
    if words:
        norm_target = r"\s+".join(re.escape(w) for w in words)
        m_norm = re.search(norm_target, raw_text, re.IGNORECASE)
        if m_norm:
            return m_norm.start(), m_norm.end()

    # 4. Fallback: bounded window search around canon_start
    first_word = words[0] if words else ""
    last_word = words[-1] if words else ""
    if first_word and last_word:
        window_start = max(0, canon_start - 120)
        window_end = min(len(raw_text), canon_end + 120)
        window_text = raw_text[window_start:window_end]

        m_first = re.search(re.escape(first_word), window_text, re.IGNORECASE)
        if m_first:
            start_pos = window_start + m_first.start()
            remaining = raw_text[start_pos:window_end]
            m_last = list(re.finditer(re.escape(last_word), remaining, re.IGNORECASE))
            if m_last:
                end_pos = start_pos + m_last[-1].end()
                return start_pos, end_pos

    return 0, len(raw_text)


def verify(canon: str, s: Sent) -> bool:
    """Verify that a sentence span matches the canonical document text byte-for-byte."""
    if s.start < 0 or s.end > len(canon) or s.start > s.end:
        return False
    return canon[s.start:s.end] == s.text


def init_punkt() -> PunktSentenceTokenizer:
    """Initialize NLTK Punkt tokenizer with scientific abbreviations."""
    global _punkt_tokenizer
    if _punkt_tokenizer is None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        params = PunktParameters()
        params.abbrev_types = SCIENTIFIC_ABBREVIATIONS
        _punkt_tokenizer = PunktSentenceTokenizer(params)
    return _punkt_tokenizer


# 1-character dot substitution (\u2024 ONE DOT LEADER) preserving exact byte/char offsets
_PROTECT_DOT = "\u2024"
_RE_DECIMAL = re.compile(r"(\d+)\.(\d+)")
_RE_ET_AL = re.compile(r"\b(et\s+al)\.", re.I)
_RE_FIG = re.compile(r"\b(fig|figs)\.\s*(\d+)", re.I)
_RE_GENUS = re.compile(r"\b([A-Z])\.\s+([a-z]+)")


def split_sentences_with_spans(canon: str, source: str = "abstract") -> List[Sent]:
    """
    Split canonical text into sentence spans using protected tokenization.
    Guarantees canon[sent.start:sent.end] == sent.text for all returned spans.
    """
    if not canon:
        return []
    tokenizer = init_punkt()

    # Protect vulnerable dot patterns with 1-character substitution so lengths stay identical
    protected = canon
    protected = _RE_DECIMAL.sub(rf"\1{_PROTECT_DOT}\2", protected)
    protected = _RE_ET_AL.sub(rf"\1{_PROTECT_DOT}", protected)
    protected = _RE_FIG.sub(rf"\1{_PROTECT_DOT} \2", protected)
    protected = _RE_GENUS.sub(rf"\1{_PROTECT_DOT} \2", protected)

    raw_spans = list(tokenizer.span_tokenize(protected))
    if not raw_spans:
        return []

    merged_spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(raw_spans):
        start, end = raw_spans[i]
        while i + 1 < len(raw_spans):
            next_start, next_end = raw_spans[i + 1]
            prev_slice = canon[start:end].strip()
            next_slice = canon[next_start:next_end].strip()

            words = prev_slice.rstrip(". ").split()
            last_word = words[-1].lower() if words else ""
            prev_ends_abbrev = last_word in SCIENTIFIC_ABBREVIATIONS
            next_starts_lower = len(next_slice) > 0 and (next_slice[0].islower() or next_slice[0].isdigit())

            if prev_ends_abbrev or next_starts_lower:
                end = next_end
                i += 1
            else:
                break
        merged_spans.append((start, end))
        i += 1

    sentences: List[Sent] = []
    sent_idx = 0
    for start, end in merged_spans:
        slice_text = canon[start:end]
        l_stripped = len(slice_text) - len(slice_text.lstrip())
        r_stripped = len(slice_text) - len(slice_text.rstrip())

        adj_start = start + l_stripped
        adj_end = end - r_stripped
        clean_sent_text = canon[adj_start:adj_end]

        if len(clean_sent_text) > 3 and any(c.isalnum() for c in clean_sent_text):
            sentences.append(Sent(
                text=clean_sent_text,
                start=adj_start,
                end=adj_end,
                sentence_index=sent_idx,
                source=source
            ))
            sent_idx += 1

    return sentences


def split_sentences(text: str) -> List[str]:
    """Convenience wrapper returning plain sentence strings."""
    return [s.text for s in split_sentences_with_spans(canonicalize(text))]


def build_entity_regex(entity: str, aliases: Optional[List[str]] = None) -> re.Pattern:
    """
    Build a zero-width lookaround regex matching the entity and vetted aliases.
    Uses (?<![A-Za-z0-9\\-_]) and (?![A-Za-z0-9\\-_]) boundaries.
    """
    vetted_aliases = vet_aliases(aliases, entity)
    terms: List[str] = [entity.strip()]
    for a in vetted_aliases:
        clean_a = a.strip()
        if clean_a and clean_a.lower() != entity.strip().lower() and clean_a not in terms:
            terms.append(clean_a)

    subpatterns: List[str] = []
    for term in terms:
        if not term:
            continue
        escaped = re.escape(term)
        escaped = re.sub(r"(\\\s+|\\-)", r"[- ]?", escaped)
        plural = r"(?:s|es)?" if term[-1].isalnum() else ""
        subpatterns.append(rf"(?:(?<!\w){escaped}{plural}(?!\w))")

    pattern_str = rf"(?:{'|'.join(subpatterns)})"
    return re.compile(pattern_str, re.IGNORECASE)


_DEFAULT_PRODUCTION_VERBS = [
    "synthesize", "synthesized", "synthesizes", "synthesizing", "synthesised",
    "produce", "produced", "produces", "producing",
    "afford", "afforded", "affords", "affording",
    "yield", "yielded", "yields", "yielding",
    "biosynthesize", "biosynthesized", "biosynthesizes", "biosynthesizing",
    "isolate", "isolated", "isolates", "isolating",
    "extract", "extracted", "extracts", "extracting",
    "prepare", "prepared", "prepares", "preparing",
    "ferment", "fermented", "ferments", "fermenting",
    "generate", "generated", "generates", "generating",
    "obtain", "obtained", "obtains", "obtaining",
    "engineer", "engineered", "engineers", "engineering",
    "express", "expressed", "expresses", "expressing",
    "titer", "titers", "accumulation", "overproduction"
]

_RE_COMMERCIAL_OBTAIN = re.compile(
    r"\b(?:obtained|purchased)\s+(?:from|commercially)\b",
    re.I
)

_RE_REVIEW_ARTICLE = re.compile(
    r"\b(?:review|systematic review|meta-analysis|overview|perspective|minireview)\b|"
    r"we\s+synthesize\s+(?:the\s+|current\s+)?(?:knowledge|literature|findings|data|evidence)",
    re.I
)

_RE_THIS_WORK_POSITIVE = re.compile(
    r"\b(?:we\s+(?:synthesized|prepared|produced|engineered|isolated|report|demonstrate|show|developed)|"
    r"was\s+(?:obtained|synthesized|prepared|produced|afforded|yielded|isolated)|"
    r"were\s+(?:obtained|synthesized|prepared|produced|afforded|yielded|isolated)|"
    r"in\s+(?:this|the\s+present)\s+study|herein|in\s+this\s+work|our\s+results)\b",
    re.I
)

_RE_THIS_WORK_BACKGROUND = re.compile(
    r"\b(?:has\s+been\s+(?:produced|synthesized|reported|isolated|prepared)|"
    r"have\s+been\s+(?:produced|synthesized|reported|isolated|prepared)|"
    r"reported\s+to|previously|it\s+is\s+known|prior\s+studies|earlier\s+reported|widely\s+used)\b",
    re.I
)


def has_production_verb(sentence: str, custom_verbs: Optional[List[str]] = None) -> bool:
    """
    Check whether the sentence contains a genuine production or synthesis action verb.
    Rejects commercial purchase mentions ('obtained from Sigma').
    """
    if _RE_COMMERCIAL_OBTAIN.search(sentence):
        if not re.search(r"\b(?:yield|synthesized|prepared|yielded)\b", sentence, re.I):
            return False

    verbs = custom_verbs if custom_verbs is not None else _DEFAULT_PRODUCTION_VERBS
    pattern = r"\b(?:" + "|".join(re.escape(v) for v in verbs) + r")\b"
    return bool(re.search(pattern, sentence, re.I))


def is_review_article(title: Optional[str], text: Optional[str]) -> bool:
    """Check if the title or abstract is explicitly a review or synthesis of knowledge."""
    if title and _RE_REVIEW_ARTICLE.search(title):
        return True
    if text and _RE_REVIEW_ARTICLE.search(text):
        return True
    return False


def detect_this_work_cues(sentence: str) -> Tuple[bool, bool]:
    """
    Detect experimental first-person cues vs. background/citation cues in sentence.
    Returns: (has_positive_cue, has_background_cue)
    """
    has_pos = bool(_RE_THIS_WORK_POSITIVE.search(sentence))
    has_bg = bool(_RE_THIS_WORK_BACKGROUND.search(sentence))
    return has_pos, has_bg


# Backward-compatibility aliases for legacy test fixtures
clean_text = canonicalize


def candidate_sentences(text: str, entity: str) -> List[str]:
    reg = build_entity_regex(entity)
    return [s.text for s in split_sentences_with_spans(canonicalize(text)) if reg.search(s.text)]
