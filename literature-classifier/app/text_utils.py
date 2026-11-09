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


split_sentences_with_offsets = split_sentences_with_spans

# WS15-B & WS14-E constants (registered in config/constants.md)
CLAUSE_SPLIT_MIN = 180
ANAPHORA_WINDOW = 2

# F5: Unified chemical compound prefix list (registered in config/constants.md)
COMPOUND_PREFIXES = ["dihydro", "deoxy", "iso", "nor", "epi"]
_RELATED_PREFIXES = COMPOUND_PREFIXES

# F6: Evidence quality threshold (registered in config/constants.md)
MIN_EVIDENCE_CHARS = 30

# F6: Finite production verbs (verb-lexicon check)
FINITE_PRODUCTION_VERBS = [
    "synthesize", "synthesized", "synthesizes", "synthesizing",
    "synthesise", "synthesised", "synthesises", "synthesising",
    "produce", "produced", "produces", "producing",
    "afford", "afforded", "affords", "affording",
    "yield", "yielded", "yields", "yielding",
    "biosynthesize", "biosynthesized", "biosynthesizes", "biosynthesizing",
    "biosynthesise", "biosynthesised", "biosynthesises", "biosynthesising",
    "isolate", "isolated", "isolates", "isolating",
    "obtain", "obtained", "obtains", "obtaining",
    "prepare", "prepared", "prepares", "preparing",
    "generate", "generated", "generates", "generating",
    "construct", "constructed", "constructs", "constructing",
    "engineer", "engineered", "engineers", "engineering",
    "express", "expressed", "expresses", "expressing",
    "accumulate", "accumulated", "accumulates", "accumulating",
    "convert", "converted", "converts", "converting",
    "ferment", "fermented", "ferments", "fermenting",
    "assemble", "assembled", "assembles", "assembling",
    "form", "formed", "forms", "forming",
    "demonstrate", "demonstrated", "demonstrates",
    "show", "showed", "shown", "shows",
    "achieve", "achieved", "achieves",
    "develop", "developed", "develops",
]

# F13: Review detection beyond pubType (registered in config/constants.md)
REVIEW_STRONG_CUES = ["this review", "we review", "literature review", "here we summarize"]
REVIEW_WEAK_CUES = [
    "recent advances", "perspectives", "clinical potential", "mechanistic insights",
    "biotechnological advances", "current understanding", "future perspectives", "state of the art"
]
RESULTS_MARKERS = ["we show", "we found", "we constructed", "was achieved", "were synthesized"]


def has_finite_production_verb(text: str) -> bool:
    """Check whether text contains a finite action verb from FINITE_PRODUCTION_VERBS (F4/F6)."""
    low = text.lower()
    return any(re.search(rf"\b{re.escape(v)}\b", low) for v in FINITE_PRODUCTION_VERBS)


def detect_review_cues(title: Optional[str], text: Optional[str]) -> Tuple[bool, bool, bool]:
    """
    Evaluate text cues on title and abstract for review status (F13).
    Returns: (has_strong_cue, has_weak_cue, has_results_marker)
    Title is used for classification only, never as evidence.
    """
    combined = f"{title or ''} {text or ''}".lower()
    has_results = any(marker in combined for marker in RESULTS_MARKERS)
    has_strong = any(cue in combined for cue in REVIEW_STRONG_CUES)
    has_weak = any(cue in combined for cue in REVIEW_WEAK_CUES)
    return has_strong, has_weak, has_results


def is_related_compound_collision(sentence: str, entity: str) -> bool:
    """Detect if entity match in sentence is actually a modified related chemical derivative (F5)."""
    clean_ent = entity.strip().lower()
    for prefix in COMPOUND_PREFIXES:
        pattern = rf"\b{prefix}[- ]?{re.escape(clean_ent)}\b"
        pattern_concat = rf"\b{prefix}{re.escape(clean_ent)}\b"
        if re.search(pattern, sentence, re.IGNORECASE) or re.search(pattern_concat, sentence, re.IGNORECASE):
            standalone = rf"(?<!\w){re.escape(clean_ent)}(?!\w)"
            matches = list(re.finditer(standalone, sentence, re.IGNORECASE))
            if not matches or all(
                (m.start() >= len(prefix) and sentence[m.start() - len(prefix):m.start()].lower().startswith(prefix))
                for m in matches
            ):
                return True
    return False

_CONJUNCTION_MARKERS = (", however", ", whereas", ", although", ", but ")
_ANAPHORA_RE = re.compile(
    r"\b(?:this|the(?:\s+(?:resulting|target|same|above))?)\s+"
    r"(?:compound|product|molecule|alkaloid|metabolite|substance|entity|ester|acid)\b|"
    r"\bthe\s+former\b|"
    r"(?<!\w)it(?!\w)(?!\s+(?:is|was|remains|has\s+been)\s+(?:known|clear|important|noteworthy|widely|reported|shown|found|suggested))",
    re.IGNORECASE,
)


def _find_depth0_splits(text: str, mode: str) -> List[Tuple[int, int]]:
    """
    Scan text tracking bracket nesting depth across '([{' and ')]}'.
    Returns list of (cut_end_of_left, start_of_right) at bracket depth 0.
    """
    splits: List[Tuple[int, int]] = []
    depth = 0
    lower = text.lower()
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            if mode == "semicolon" and ch == ";":
                splits.append((i, i + 1))
            elif mode == "conjunction" and ch == ",":
                matched_marker = None
                for marker in _CONJUNCTION_MARKERS:
                    if lower.startswith(marker, i):
                        # Require word boundary / comma / space after marker word
                        end_idx = i + len(marker)
                        if end_idx >= n or not text[end_idx].isalnum():
                            matched_marker = marker
                            break
                if matched_marker is not None:
                    # Keep the conjunction word ('however', 'but', etc.) on the right clause
                    splits.append((i, i + 1))
            elif mode == "comma" and ch == "," and i + 1 < n and text[i + 1] == " ":
                splits.append((i, i + 1))
        i += 1
    return splits


def _apply_splits_to_span(sent: Sent, splits: List[Tuple[int, int]]) -> List[Sent]:
    """Slice a Sent at (left_end, right_start) pairs while preserving byte-exact offsets."""
    if not splits:
        return [sent]
    pieces: List[Sent] = []
    cursor = 0
    text = sent.text
    for left_end, right_start in splits + [(len(text), len(text))]:
        raw_piece = text[cursor:left_end]
        l_strip = len(raw_piece) - len(raw_piece.lstrip())
        r_strip = len(raw_piece) - len(raw_piece.rstrip())
        trimmed = raw_piece.strip()
        if trimmed and len(trimmed) >= 4:
            p_start = sent.start + cursor + l_strip
            p_end = sent.start + left_end - r_strip
            pieces.append(Sent(
                text=trimmed,
                start=p_start,
                end=p_end,
                sentence_index=sent.sentence_index,
                source=sent.source,
            ))
        cursor = right_start
    return pieces if pieces else [sent]


def split_clauses_bracket_aware(sent: Sent, min_len: int = CLAUSE_SPLIT_MIN) -> List[Sent]:
    """
    Bracket-depth-aware syntactic clause splitter (WS15-B / WS17-R5).
    Only splits at bracket depth 0 so IUPAC chemical names like '2,2'-bipyridine',
    'N,N-dimethyltryptamine', and 'dibenzo[b,d]oxepine' stay intact.
    Priority order: ';' -> ', however/whereas/although/but' -> ', ' fallback.
    """
    if len(sent.text) < min_len:
        return [sent]

    # Step 1: semicolon splits (depth 0)
    clauses = _apply_splits_to_span(sent, _find_depth0_splits(sent.text, "semicolon"))

    # Step 2: conjunction splits for any clause still >= min_len
    next_clauses: List[Sent] = []
    for cl in clauses:
        if len(cl.text) >= min_len:
            next_clauses.extend(_apply_splits_to_span(cl, _find_depth0_splits(cl.text, "conjunction")))
        else:
            next_clauses.append(cl)
    clauses = next_clauses

    # Step 3: depth-0 ', ' fallback for any clause still >= min_len (min_len=0 forces all levels when requested)
    final_clauses: List[Sent] = []
    for cl in clauses:
        if len(cl.text) >= min_len and len(cl.text) > 20:
            # Skip splitting tiny introductory phrases unless still over min_len
            comma_splits = _find_depth0_splits(cl.text, "comma")
            split_parts = _apply_splits_to_span(cl, comma_splits)
            # Only accept comma fallback if fragments are non-trivial (>= 12 chars each)
            if all(len(p.text) >= 12 for p in split_parts):
                final_clauses.extend(split_parts)
            else:
                final_clauses.append(cl)
        else:
            final_clauses.append(cl)

    return final_clauses


def resolve_anaphora(
    sentences: List[Sent],
    entity_regex: re.Pattern,
    window: int = ANAPHORA_WINDOW,
) -> Dict[int, int]:
    """
    Deterministic, zero-ML cross-sentence anaphora resolver (WS14-E).
    Maps sentence index i (lacking literal entity match but containing chemical anaphora)
    to nearest antecedent sentence index j in [i - window, i - 1] that matches entity_regex.
    Strictly no lookahead.
    """
    resolved: Dict[int, int] = {}
    for idx, sent in enumerate(sentences):
        if entity_regex.search(sent.text):
            continue
        if not _ANAPHORA_RE.search(sent.text):
            continue
        start_back = max(0, idx - window)
        for back in range(idx - 1, start_back - 1, -1):
            if entity_regex.search(sentences[back].text):
                resolved[idx] = back
                break
    return resolved


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
    "synthesize", "synthesized", "synthesizes", "synthesizing", "synthesised", "synthesis",
    "produce", "produced", "produces", "producing", "production",
    "afford", "afforded", "affords", "affording",
    "yield", "yielded", "yields", "yielding",
    "biosynthesize", "biosynthesized", "biosynthesizes", "biosynthesizing", "biosynthesis",
    "isolate", "isolated", "isolates", "isolating",
    "extract", "extracted", "extracts", "extracting",
    "prepare", "prepared", "prepares", "preparing",
    "ferment", "fermented", "ferments", "fermenting", "fermentation",
    "generate", "generated", "generates", "generating",
    "obtain", "obtained", "obtains", "obtaining",
    "engineer", "engineered", "engineers", "engineering",
    "express", "expressed", "expresses", "expressing", "expression", "overexpression",
    "secretion", "secreted", "secrete", "cloned",
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
    r"\b(?:has\s+been\s+(?:produced|synthesized|reported|isolated|prepared|achieved|demonstrated|established|described|detailed)|"
    r"have\s+been\s+(?:produced|synthesized|reported|isolated|prepared|achieved|demonstrated|established|described|detailed)|"
    r"reported\s+to|previously|it\s+is\s+known|prior\s+studies|earlier\s+reported|widely\s+used|"
    r"seminal\s+(?:study|work|report|paper)|et\s+al\.?)\b",
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
