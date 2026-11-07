"""Resilient literature retrieval from Europe PMC REST API, XML parser, deduplication, and PubChem."""
import os
import re
import json
import time
import logging
import hashlib
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple, Set
import xml.etree.ElementTree as ET
import requests
from .text_utils import clean_markup_and_unescape, vet_aliases, compute_sha256

logger = logging.getLogger(__name__)

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUROPE_PMC_SEARCH_ENDPOINT = f"{EUROPE_PMC_BASE}/search"

TARGET_SECTIONS = re.compile(
    r"\b(method|material|experimental|result|discussion|procedure|synthesis|production)",
    re.I
)
EXCLUDE_SECTIONS = re.compile(
    r"\b(reference|acknowledg|funding|conflict|supplementary|author|contribution)",
    re.I
)


class RetrievalError(Exception):
    """Raised when Europe PMC fails, times out, or returns an unresolvable error."""
    pass


def escape_lucene(term: str) -> str:
    """Escape special characters that disrupt Europe PMC Lucene/Solr query parsing."""
    clean = re.sub(r'["\\]', "", term).strip()
    return clean


def normalize_title_for_dedupe(title: Optional[str]) -> str:
    """Normalize a title string for deduplication."""
    if not title:
        return ""
    clean = clean_markup_and_unescape(title).lower()
    # Remove punctuation and normalize spaces
    clean = re.sub(r"[^\w\s]", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


def is_preprint(paper: Dict[str, Any]) -> bool:
    """Check if a publication appears to be an unpeer-reviewed preprint."""
    title = (paper.get("title") or "").lower()
    doi = (paper.get("doi") or "").lower()
    pmid = paper.get("pmid")

    # If it has a verified PubMed ID (PMID), it is an indexed peer-reviewed record
    if pmid and not ("10.1101" in doi or "biorxiv" in doi or "medrxiv" in doi):
        return False

    if (
        "10.1101" in doi       # bioRxiv, medRxiv
        or "10.21203" in doi   # Research Square
        or "10.20944" in doi   # Preprints.org
        or "10.26434" in doi   # ChemRxiv
        or "biorxiv" in doi
        or "medrxiv" in doi
        or "chemrxiv" in doi
        or "preprints" in doi
        or "arxiv" in doi
        or "biorxiv" in title
        or "medrxiv" in title
        or "chemrxiv" in title
    ):
        return True

    return pmid is None


def deduplicate_papers(papers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Deduplicate publications by DOI and normalized title.
    Prefers peer-reviewed/published version over preprint.
    Attaches 'dedup_of' identifier to the surviving publication.
    Preserves and unifies 'retrieved_by' ('intent', 'plain', or 'both').
    Returns: (unique_papers, count_of_deduped_records)
    """
    unique_papers: List[Dict[str, Any]] = []
    seen_dois: Dict[str, int] = {}       # doi_lower -> index in unique_papers
    seen_titles: Dict[str, int] = {}     # norm_title -> index in unique_papers
    dedup_count = 0

    for paper in papers:
        doi = (paper.get("doi") or "").strip().lower()
        title = normalize_title_for_dedupe(paper.get("title"))
        paper_id = paper.get("pmid") or paper.get("doi") or paper.get("title") or "unknown"
        curr_retrieved_by = paper.get("retrieved_by", "intent")

        matched_idx: Optional[int] = None
        if doi and doi in seen_dois:
            matched_idx = seen_dois[doi]
        elif title and len(title) > 20 and title in seen_titles:
            matched_idx = seen_titles[title]

        if matched_idx is not None:
            existing = unique_papers[matched_idx]
            existing_id = existing.get("pmid") or existing.get("doi") or "unknown"
            dedup_count += 1

            # Update retrieved_by
            existing_retrieved = existing.get("retrieved_by", "intent")
            if existing_retrieved != curr_retrieved_by:
                existing["retrieved_by"] = "both"

            # Check if current paper is preferred (e.g. published version vs existing preprint)
            existing_is_prep = is_preprint(existing)
            curr_is_prep = is_preprint(paper)

            if existing_is_prep and not curr_is_prep:
                # Replace preprint with peer-reviewed publication
                paper_copy = dict(paper)
                paper_copy["dedup_of"] = str(existing_id)
                paper_copy["retrieved_by"] = existing.get("retrieved_by", "both")
                unique_papers[matched_idx] = paper_copy
                if doi:
                    seen_dois[doi] = matched_idx
                if title:
                    seen_titles[title] = matched_idx
            else:
                # Keep existing record and mark absorbed ID
                if not existing.get("dedup_of"):
                    existing["dedup_of"] = str(paper_id)
        else:
            paper_copy = dict(paper)
            if "dedup_of" not in paper_copy:
                paper_copy["dedup_of"] = None
            if "retrieved_by" not in paper_copy:
                paper_copy["retrieved_by"] = curr_retrieved_by

            idx = len(unique_papers)
            unique_papers.append(paper_copy)
            if doi:
                seen_dois[doi] = idx
            if title:
                seen_titles[title] = idx

    return unique_papers, dedup_count


def fetch_pubchem_synonyms(entity_name: str, timeout: float = 5.0) -> List[str]:
    """Fetch validated synonyms for an entity from PubChem PUG REST API."""
    if os.getenv("USE_STUB_SCORER", "0") == "1":
        return []

    clean_name = entity_name.strip()
    if not clean_name:
        return []

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "snapshots")
    cache_path = os.path.join(cache_dir, "pubchem_cache.json")
    cache: Dict[str, List[str]] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if clean_name.lower() in cache:
                    return cache[clean_name.lower()]
        except Exception:
            pass

    encoded_name = urllib.parse.quote(clean_name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/synonyms/JSON"

    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            raw_syns = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
            valid_syns: List[str] = []
            for s in raw_syns:
                s_clean = s.strip()
                if (
                    len(s_clean) >= 5
                    and not s_clean.isdigit()
                    and not any(char in s_clean for char in ["|", "[", "]", "{", "}", "^", "~"])
                    and s_clean.lower() != clean_name.lower()
                    and s_clean not in valid_syns
                ):
                    valid_syns.append(s_clean)
                if len(valid_syns) >= 8:
                    break

            vetted = vet_aliases(valid_syns, clean_name)
            cache[clean_name.lower()] = vetted
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            except Exception:
                pass
            return vetted
    except Exception as e:
        logger.debug(f"PubChem synonym fetch failed for '{clean_name}': {e}")

    return []


def load_snapshot_papers(entity: str) -> Optional[List[Dict[str, Any]]]:
    """Load cached snapshot papers from data/snapshots/{entity}.json if available."""
    clean_ent = entity.strip().lower().replace(" ", "_")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    snap_file = os.path.join(base_dir, "data", "snapshots", f"{clean_ent}.json")
    if os.path.exists(snap_file):
        try:
            with open(snap_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("papers", [])
        except Exception as e:
            logger.warning(f"Failed to read snapshot {snap_file}: {e}")
    return None


def search_europe_pmc(
    query: str,
    max_results: int = 15,
    timeout: float = 15.0,
    include_fulltext: bool = False,
    offline_fallback_entity: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve publications from Europe PMC REST API with retry, backoff, and Retry-After support."""
    params = {
        "query": query,
        "resultType": "core",
        "format": "json",
        "pageSize": max_results,
    }

    headers = {
        "User-Agent": "LoopwormLiteratureClassifier/3.0 (mailto:ruhaankakar@gmail.com) Python-requests"
    }

    response = None
    backoffs = [1.0, 2.0, 4.0]
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = requests.get(EUROPE_PMC_SEARCH_ENDPOINT, params=params, headers=headers, timeout=timeout)
            if response.status_code == 200:
                break

            retry_after = response.headers.get("Retry-After")
            sleep_time = backoffs[attempt]
            if retry_after and retry_after.isdigit():
                sleep_time = min(float(retry_after), 10.0)

            if response.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                logger.warning(f"Europe PMC returned HTTP {response.status_code}, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            else:
                logger.warning(f"Europe PMC returned HTTP {response.status_code}: {response.text[:120]}")
                break
        except requests.RequestException as exc:
            if attempt < max_attempts - 1:
                logger.warning(f"Europe PMC network error: {exc}. Retrying in {backoffs[attempt]}s...")
                time.sleep(backoffs[attempt])
                continue
            logger.warning(f"Europe PMC request failed after {max_attempts} attempts: {exc}")
            break

    if response is not None and response.status_code == 200:
        try:
            data = response.json()
            result_list = data.get("resultList", {}).get("result", [])
            publications: List[Dict[str, Any]] = []

            for item in result_list:
                raw_title = item.get("title")
                raw_abstract = item.get("abstractText")
                doi = item.get("doi")
                pmid = item.get("pmid")
                pmcid = item.get("pmcid")
                pub_year = item.get("pubYear")
                pub_type = item.get("pubType", "")

                clean_title = clean_markup_and_unescape(str(raw_title)) if raw_title is not None else None
                clean_abstract = clean_markup_and_unescape(str(raw_abstract)) if raw_abstract is not None else None

                raw_record_str = f"{raw_title or ''} {raw_abstract or ''}"
                record_sha = compute_sha256(raw_record_str)

                fulltext_sections = None
                if include_fulltext and pmcid:
                    xml_text = fetch_fulltext_xml(pmcid)
                    if xml_text:
                        fulltext_sections = extract_fulltext_sections(xml_text)

                publications.append({
                    "title": clean_title,
                    "doi": str(doi) if doi is not None else None,
                    "pmid": str(pmid) if pmid is not None else None,
                    "pmcid": str(pmcid) if pmcid is not None else None,
                    "abstractText": clean_abstract,
                    "raw_text": raw_record_str,
                    "source_record_sha256": record_sha,
                    "pubYear": str(pub_year) if pub_year is not None else None,
                    "pubType": pub_type,
                    "fulltext_sections": fulltext_sections,
                    "retrieved_by": "intent"
                })
            return publications
        except Exception as e:
            logger.warning(f"Europe PMC JSON parse failed: {e}")

    if offline_fallback_entity:
        snapshot_papers = load_snapshot_papers(offline_fallback_entity)
        if snapshot_papers:
            logger.info(f"Loaded {len(snapshot_papers)} papers from offline snapshot for '{offline_fallback_entity}'")
            cleaned_snapshots = []
            for sp in snapshot_papers[:max_results]:
                sp_clean = dict(sp)
                if sp_clean.get("title"):
                    sp_clean["title"] = clean_markup_and_unescape(sp_clean["title"])
                if sp_clean.get("abstractText"):
                    sp_clean["abstractText"] = clean_markup_and_unescape(sp_clean["abstractText"])
                raw_str = f"{sp_clean.get('title') or ''} {sp_clean.get('abstractText') or ''}"
                sp_clean["raw_text"] = raw_str
                sp_clean["source_record_sha256"] = compute_sha256(raw_str)
                sp_clean["retrieved_by"] = "intent"
                cleaned_snapshots.append(sp_clean)
            return cleaned_snapshots

    if response is not None and response.status_code in (429, 502, 503, 504):
        raise RetrievalError(f"Europe PMC temporary outage (HTTP {response.status_code}).")

    raise RetrievalError("Failed to retrieve response from Europe PMC.")


def search_europe_pmc_dual(
    query_intent: str,
    query_plain: str,
    max_results: int = 15,
    timeout: float = 15.0,
    include_fulltext: bool = False,
    offline_fallback_entity: Optional[str] = None,
    search_fn: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Dual Query Retrieval (Phase 8.1):
    Unions intent-augmented query with plain entity query, tags each record with
    retrieved_by ('intent', 'plain', or 'both'), and deduplicates.
    Returns: (deduplicated_papers, dedup_count)
    """
    fn = search_fn or search_europe_pmc

    try:
        intent_results = fn(
            query=query_intent,
            max_results=max_results,
            timeout=timeout,
            include_fulltext=include_fulltext,
            offline_fallback_entity=offline_fallback_entity
        )
        for r in intent_results:
            r["retrieved_by"] = "intent"
    except RetrievalError:
        raise
    except Exception as exc:
        logger.warning(f"Intent query retrieval failed: {exc}")
        intent_results = []

    try:
        plain_results = fn(
            query=query_plain,
            max_results=max_results,
            timeout=timeout,
            include_fulltext=include_fulltext,
            offline_fallback_entity=offline_fallback_entity
        )
        for r in plain_results:
            r["retrieved_by"] = "plain"
    except RetrievalError:
        raise
    except Exception as exc:
        logger.warning(f"Plain query retrieval failed: {exc}")
        plain_results = []

    # If both failed and offline fallback was requested
    if not intent_results and not plain_results and offline_fallback_entity:
        intent_results = fn(
            query=query_intent,
            max_results=max_results,
            offline_fallback_entity=offline_fallback_entity
        )

    # Combine and deduplicate
    combined = intent_results + plain_results
    unique_papers, dedup_count = deduplicate_papers(combined)
    return unique_papers[:max_results], dedup_count


def fetch_fulltext_xml(pmcid: str, timeout: float = 10.0) -> Optional[str]:
    """Fetch full-text XML for open-access papers with PMCID from Europe PMC."""
    if not pmcid:
        return None
    clean_id = pmcid.strip()
    if not clean_id.upper().startswith("PMC"):
        clean_id = f"PMC{clean_id}"

    url = f"{EUROPE_PMC_BASE}/{clean_id}/fullTextXML"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text:
            return r.text
        return None
    except requests.RequestException:
        logger.debug(f"Full-text XML request failed or timed out for {clean_id}")
        return None


def extract_fulltext_sections(xml_text: str) -> List[Tuple[str, str]]:
    """Extracts text paragraphs from target sections of full-text XML."""
    if not xml_text:
        return []

    sections: List[Tuple[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
        for tag in ("xref", "table-wrap", "fig", "ref-list"):
            for elem in root.findall(f".//{tag}"):
                elem.clear()

        for sec in root.iter("sec"):
            sec_type = sec.attrib.get("sec-type", "")
            title_node = sec.find("title")
            title_text = "".join(title_node.itertext()).strip() if title_node is not None else ""

            combined_sec = f"{sec_type} {title_text}"
            if EXCLUDE_SECTIONS.search(combined_sec):
                continue

            is_target = bool(TARGET_SECTIONS.search(combined_sec))
            if is_target or not (sec_type or title_text):
                tag_name = sec_type if sec_type else (title_text[:20] if title_text else "body")
                for p in sec.iter("p"):
                    p_text = "".join(p.itertext()).strip()
                    if p_text and len(p_text) > 20:
                        clean_p = clean_markup_and_unescape(p_text)
                        sections.append((clean_p, tag_name))
    except Exception as exc:
        logger.debug(f"XML parse error in fulltext extraction: {exc}")

    return sections
