import re
import html
import logging
import statistics
from typing import List, Dict, Any, Optional, Tuple
from app.nli import NLIScorer
from app.text_utils import candidate_sentences, build_entity_regex, clean_text, split_sentences
from app.paraphrase import normalise_criterion, generate_criterion_paraphrases, is_synthesis_criterion
from app.scoring import (
    calculate_qualify_score,
    calculate_confidence,
    get_confidence_band,
    rank_papers,
)

logger = logging.getLogger(__name__)

# Default tuned decision threshold tau and contrast guard delta
DEFAULT_THRESHOLD = 0.50
DEFAULT_CONTRAST_DELTA = 0.10

# Affirmative contrast hypotheses avoiding trailing negative-clause penalization in MNLI
DEFAULT_CONTRAST_HYPOTHESES = [
    "{entity} is a starting material or reactant in this reaction.",
    "{entity} was used as a reagent, solvent, or analytical standard in this study.",
    "{entity} was administered, consumed, or given to subjects or participants in this study.",
]


class EvidenceIntegrityError(Exception):
    """Raised when an extracted evidence sentence cannot be verified as a verbatim substring of source text."""
    pass


def _assert_verbatim(evidence: Optional[str], pub: Dict[str, Any]):
    """
    Guarantees the 'no invented evidence' requirement.
    Checks that evidence is an exact substring of title, abstract, or fulltext sections.
    """
    if evidence is None:
        return

    sources = []
    if pub.get("title"):
        sources.append(clean_text(pub["title"]))
    if pub.get("abstractText"):
        sources.append(clean_text(pub["abstractText"]))
    if pub.get("fulltext_sections"):
        for sec in pub["fulltext_sections"]:
            sources.append(clean_text(sec))

    clean_ev = clean_text(evidence)
    matched = any(clean_ev in src for src in sources)

    if not matched:
        # Fallback check on raw uncleaned texts with whitespace collapsed
        def raw_norm(t: str) -> str:
            return re.sub(r'\s+', ' ', html.unescape(t)).strip()

        norm_ev = raw_norm(evidence)
        raw_sources = []
        if pub.get("title"):
            raw_sources.append(raw_norm(str(pub["title"])))
        if pub.get("abstractText"):
            raw_sources.append(raw_norm(str(pub["abstractText"])))
        if pub.get("fulltext_sections"):
            for sec in pub["fulltext_sections"]:
                raw_sources.append(raw_norm(str(sec)))

        matched = any(norm_ev in rsrc for rsrc in raw_sources)

    if not matched:
        raise EvidenceIntegrityError(
            f"Evidence '{evidence}' is not a verbatim substring of publication '{pub.get('pmid') or pub.get('title')}'."
        )


def format_contrast_hypotheses(
    custom_hypotheses: Optional[List[str]],
    legacy_template: Optional[str],
    criterion: str,
    entity: str
) -> Tuple[List[str], str]:
    """
    Determines and formats the active contrast hypotheses:
    - If custom_hypotheses provided: formats each with entity substitution ('custom').
    - If legacy_template provided: formats single template ('custom').
    - Otherwise, checks is_synthesis_criterion(criterion):
      - If True: uses DEFAULT_CONTRAST_HYPOTHESES ('active').
      - If False: returns empty list ('not_applicable') to avoid false contrast firing on non-synthesis tasks.
    """
    clean_ent = entity.strip()
    if custom_hypotheses:
        formatted = []
        for h in custom_hypotheses:
            s = h.strip()
            if "{entity}" in s:
                s = s.replace("{entity}", clean_ent)
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
            formatted.append(s)
        return formatted, "custom"

    if legacy_template:
        s = legacy_template.strip()
        if "{entity}" in s:
            s = s.replace("{entity}", clean_ent)
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        return [s], "custom"

    if is_synthesis_criterion(criterion):
        formatted = []
        for h in DEFAULT_CONTRAST_HYPOTHESES:
            s = h.replace("{entity}", clean_ent)
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
            formatted.append(s)
        return formatted, "active"

    return [], "not_applicable"


def format_contrast_hypothesis(template: Optional[str], entity: str) -> str:
    """Legacy helper for single contrast template."""
    hyps, _ = format_contrast_hypotheses(None, template, "synthesis", entity)
    return hyps[0] if hyps else ""



class ClassificationRunResult(tuple):
    """
    Custom 4-tuple subclass that supports:
    1) Backwards-compatible unpacking: results, norm_criterion, tau, delta = classify_publications(...)
    2) Named property access: out.hypotheses_used, out.contrast_guard_active
    """
    def __new__(
        cls,
        results: List[Dict[str, Any]],
        normalised_criterion: str,
        tau: float,
        delta: float,
        hypotheses_used: Optional[List[str]] = None,
        contrast_guard_active: str = "active"
    ):
        instance = super().__new__(cls, (results, normalised_criterion, tau, delta))
        instance.results = results
        instance.normalised_criterion = normalised_criterion
        instance.tau = tau
        instance.delta = delta
        instance.hypotheses_used = hypotheses_used or []
        instance.contrast_guard_active = contrast_guard_active
        return instance


def classify_publications(
    publications: List[Dict[str, Any]],
    entity: str,
    criterion: str,
    synonyms: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    contrast_delta: Optional[float] = None,
    contrast_template: Optional[str] = None,
    contrast_hypotheses: Optional[List[str]] = None,
    scorer: Optional[NLIScorer] = None,
    max_candidates: int = 10,
) -> ClassificationRunResult:
    """
    Classifies a list of Europe PMC publication dicts against the criterion.

    Returns:
      ClassificationRunResult((ranked_results, normalised_criterion, tau, delta))
      with .hypotheses_used and .contrast_guard_active properties.
    """
    if scorer is None:
        raise ValueError("An NLIScorer instance must be provided to classify_publications.")

    tau = threshold if threshold is not None else DEFAULT_THRESHOLD
    delta = contrast_delta if contrast_delta is not None else DEFAULT_CONTRAST_DELTA

    norm_criterion = normalise_criterion(criterion, entity)
    criterion_paraphrases = generate_criterion_paraphrases(criterion, entity)
    contrast_hyps, contrast_guard_active = format_contrast_hypotheses(
        custom_hypotheses=contrast_hypotheses,
        legacy_template=contrast_template,
        criterion=criterion,
        entity=entity
    )
    entity_regex = build_entity_regex(entity, synonyms)

    all_hypotheses_used = list(criterion_paraphrases) + list(contrast_hyps)

    # 1. Candidate sentence extraction per paper
    paper_candidates_meta: List[Dict[str, Any]] = []
    all_pairs_to_score: List[Tuple[str, str]] = []
    paper_pair_spans: List[Optional[Tuple[int, int]]] = []

    for pub in publications:
        title = pub.get("title")
        abstract = pub.get("abstractText")
        fulltext_sections = pub.get("fulltext_sections") or []

        candidates, title_has_entity, n_entity_sents = candidate_sentences(
            title=title,
            abstract=abstract,
            entity_regex=entity_regex,
            max_candidates=max_candidates
        )

        cand_sources: Dict[str, str] = {c: "abstract" for c in candidates}

        # Full-text XML candidate extraction fallback if available
        if fulltext_sections and len(candidates) < max_candidates:
            for sec in fulltext_sections:
                sec_clean = clean_text(sec)
                if not entity_regex.search(sec_clean):
                    continue
                sec_sents = split_sentences(sec_clean)
                for s in sec_sents:
                    if entity_regex.search(s) and s not in candidates:
                        candidates.append(s)
                        cand_sources[s] = "fulltext"
                        n_entity_sents += 1
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break

        meta = {
            "pub": pub,
            "candidates": candidates,
            "cand_sources": cand_sources,
            "title_has_entity": title_has_entity,
            "n_entity_sentences": n_entity_sents,
        }
        paper_candidates_meta.append(meta)

        if not candidates:
            paper_pair_spans.append(None)
        else:
            start_idx = len(all_pairs_to_score)
            for sent in candidates:
                # Add (sent, cp) for each criterion paraphrase
                for cp in criterion_paraphrases:
                    all_pairs_to_score.append((sent, cp))
                # Add (sent, ch) for each active contrast hypothesis
                for ch in contrast_hyps:
                    all_pairs_to_score.append((sent, ch))
            end_idx = len(all_pairs_to_score)
            paper_pair_spans.append((start_idx, end_idx))

    # 2. Batched NLI inference across all pairs
    scored_pairs = scorer.score(all_pairs_to_score) if all_pairs_to_score else []

    # 3. Decision logic per paper
    results: List[Dict[str, Any]] = []
    n_crit = len(criterion_paraphrases)
    n_cont = len(contrast_hyps)
    pairs_per_cand = n_crit + n_cont

    for idx, meta in enumerate(paper_candidates_meta):
        pub = meta["pub"]
        candidates = meta["candidates"]
        cand_sources = meta["cand_sources"]
        title_has_entity = meta["title_has_entity"]
        n_entity_sents = meta["n_entity_sentences"]
        span = paper_pair_spans[idx]

        if span is None or not candidates:
            # Case: No usable text
            res = {
                "title": pub.get("title"),
                "pmid": pub.get("pmid"),
                "doi": pub.get("doi"),
                "pmcid": pub.get("pmcid"),
                "classification": "does_not_qualify",
                "evidence": None,
                "evidence_status": "insufficient_text",
                "evidence_source": "abstract",
                "qualify_score": None,
                "confidence": None,
                "confidence_band": "insufficient",
                "title_has_entity": False,
                "n_entity_sentences": 0,
            }
            results.append(res)
            continue

        start_idx, end_idx = span
        paper_scores = scored_pairs[start_idx:end_idx]

        sentence_margins: List[float] = []
        sentence_p_ents: List[float] = []
        contrast_p_ents: List[float] = []

        for c_i in range(len(candidates)):
            cand_offset = c_i * pairs_per_cand
            cand_crit_scores = paper_scores[cand_offset : cand_offset + n_crit]
            cand_cont_scores = paper_scores[cand_offset + n_crit : cand_offset + pairs_per_cand]

            cand_margins = [crit[0] - crit[2] for crit in cand_crit_scores]
            cand_p_ents = [crit[0] for crit in cand_crit_scores]

            # Paraphrase ensemble median aggregation
            m_i = statistics.median(cand_margins)
            p_ent_i = statistics.median(cand_p_ents)

            # Max entailment over contrast hypotheses
            c_i = max((cont[0] for cont in cand_cont_scores), default=0.0)

            sentence_margins.append(m_i)
            sentence_p_ents.append(p_ent_i)
            contrast_p_ents.append(c_i)

        # Candidate with highest margin
        best_idx = max(range(len(sentence_margins)), key=lambda k: sentence_margins[k])
        m_best = sentence_margins[best_idx]
        p_ent_best = sentence_p_ents[best_idx]
        c_best = contrast_p_ents[best_idx]

        # Sentence where contrast scored highest
        highest_contrast_idx = max(range(len(contrast_p_ents)), key=lambda k: contrast_p_ents[k])

        # Runner-up candidate margin gap for confidence band
        if len(sentence_margins) > 1:
            sorted_margins = sorted(sentence_margins, reverse=True)
            margin_gap = sorted_margins[0] - sorted_margins[1]
        else:
            margin_gap = None

        # Contrast guard check
        contrast_guard_fired = (
            contrast_guard_active != "not_applicable"
            and (c_best > (p_ent_best + delta))
        )

        # Decision rule: qualifies if m_best >= tau and not contrast_guard_fired
        qualifies = (m_best >= tau) and (not contrast_guard_fired)

        # Evidence selection state machine
        if qualifies:
            classification = "qualifies"
            evidence = candidates[best_idx]
            evidence_status = "entailing_sentence"
            conf = calculate_confidence(
                m=m_best,
                tau=tau,
                is_qualifies=True,
                contrast_flipped=False
            )
        else:
            classification = "does_not_qualify"
            contrast_flipped = (m_best >= tau) and contrast_guard_fired

            if contrast_guard_fired:
                evidence = candidates[highest_contrast_idx]
                evidence_status = "contrast_mention"
            else:
                evidence = candidates[highest_contrast_idx]
                evidence_status = "strongest_mention"

            conf = calculate_confidence(
                m=m_best,
                tau=tau,
                is_qualifies=False,
                contrast_flipped=contrast_flipped,
                c_best=c_best
            )

        qualify_score = calculate_qualify_score(m_best)
        confidence_band = get_confidence_band(conf, is_insufficient=False, margin_gap=margin_gap)
        evidence_source = cand_sources.get(evidence, "abstract")

        # Verbatim evidence assertion enforcing "no invented evidence"
        _assert_verbatim(evidence, pub)

        res = {
            "title": pub.get("title"),
            "pmid": pub.get("pmid"),
            "doi": pub.get("doi"),
            "pmcid": pub.get("pmcid"),
            "classification": classification,
            "evidence": evidence,
            "evidence_status": evidence_status,
            "evidence_source": evidence_source,
            "qualify_score": qualify_score,
            "confidence": conf,
            "confidence_band": confidence_band,
            "title_has_entity": title_has_entity,
            "n_entity_sentences": n_entity_sents,
        }
        results.append(res)

    # 4. Monotonic ranking
    ranked = rank_papers(results)
    return ClassificationRunResult(
        results=ranked,
        normalised_criterion=norm_criterion,
        tau=tau,
        delta=delta,
        hypotheses_used=all_hypotheses_used,
        contrast_guard_active=contrast_guard_active
    )
