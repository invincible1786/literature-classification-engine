import time
import logging
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
import requests

logger = logging.getLogger(__name__)

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUROPE_PMC_SEARCH_ENDPOINT = f"{EUROPE_PMC_BASE}/search"


class RetrievalError(Exception):
    """Raised when Europe PMC fails, times out, or returns an unresolvable error."""
    pass


def build_query(entity: str, synonyms: Optional[List[str]] = None) -> str:
    """
    Build the Europe PMC TITLE_ABS query.
    If no synonyms: TITLE_ABS:"<entity>"
    If synonyms: TITLE_ABS:("<entity>" OR "<syn1>" OR "<syn2>")
    Sanitizes double quotes to prevent malformed query syntax.
    """
    clean_entity = entity.replace('"', '').strip()
    terms = [clean_entity]
    if synonyms:
        for syn in synonyms:
            s = syn.replace('"', '').strip()
            if s and s.lower() != clean_entity.lower() and s not in terms:
                terms.append(s)

    if len(terms) == 1:
        return f'TITLE_ABS:"{terms[0]}"'
    else:
        quoted = [f'"{t}"' for t in terms]
        joined = " OR ".join(quoted)
        return f'TITLE_ABS:({joined})'


def search_europe_pmc(
    query: str,
    max_results: int = 15,
    timeout: float = 15.0,
    include_fulltext: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieve publications from Europe PMC REST API.
    Extracts title, doi, pmid, pmcid, abstractText, pubYear.
    Optionally fetches open-access full-text XML when include_fulltext=True.
    Implements retry/backoff on 429/5xx status codes and network timeouts.
    """
    params = {
        "query": query,
        "resultType": "core",
        "format": "json",
        "pageSize": max_results,
    }

    response = None
    backoffs = [1.0, 2.0, 4.0]
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = requests.get(EUROPE_PMC_SEARCH_ENDPOINT, params=params, timeout=timeout)
            if response.status_code == 200:
                break
            elif response.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                logger.warning(f"Europe PMC returned HTTP {response.status_code}, retrying in {backoffs[attempt]}s...")
                time.sleep(backoffs[attempt])
                continue
            else:
                logger.error(f"Europe PMC returned status code {response.status_code}: {response.text[:200]}")
                raise RetrievalError(f"Europe PMC returned HTTP {response.status_code}: {response.text[:200]}")
        except requests.RequestException as exc:
            if attempt < max_attempts - 1:
                logger.warning(f"Europe PMC request error: {exc}. Retrying in {backoffs[attempt]}s...")
                time.sleep(backoffs[attempt])
                continue
            logger.error(f"Europe PMC request failed after {max_attempts} attempts: {exc}")
            raise RetrievalError(f"Failed to communicate with Europe PMC: {exc}") from exc

    if response is None or response.status_code != 200:
        raise RetrievalError("Failed to retrieve response from Europe PMC.")

    try:
        data = response.json()
    except Exception as exc:
        logger.error(f"Failed to parse Europe PMC JSON response: {exc}")
        raise RetrievalError("Europe PMC returned invalid JSON") from exc

    result_list = data.get("resultList", {}).get("result", [])
    publications: List[Dict[str, Any]] = []

    for item in result_list:
        title = item.get("title")
        doi = item.get("doi")
        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        abstract_text = item.get("abstractText")
        pub_year = item.get("pubYear")

        fulltext_sections = None
        if include_fulltext and pmcid:
            xml_text = fetch_fulltext_xml(pmcid)
            if xml_text:
                fulltext_sections = extract_fulltext_sections(xml_text)

        publications.append({
            "title": str(title) if title is not None else None,
            "doi": str(doi) if doi is not None else None,
            "pmid": str(pmid) if pmid is not None else None,
            "pmcid": str(pmcid) if pmcid is not None else None,
            "abstractText": str(abstract_text) if abstract_text is not None else None,
            "pubYear": str(pub_year) if pub_year is not None else None,
            "fulltext_sections": fulltext_sections
        })

    return publications


def fetch_fulltext_xml(pmcid: str, timeout: float = 10.0) -> Optional[str]:
    """
    Fetch full-text XML for open-access papers with PMCID from Europe PMC.
    Silently returns None on 404 or non-open-access publications.
    """
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


def extract_fulltext_sections(xml_text: str) -> List[str]:
    """
    Extracts text paragraphs from Methods, Results, or Experimental sections of full-text XML.
    Falls back gracefully if XML is malformed.
    """
    if not xml_text:
        return []

    sections = []
    try:
        root = ET.fromstring(xml_text)
        # Iterate over section elements or paragraphs
        for p in root.iter("p"):
            p_text = "".join(p.itertext()).strip()
            if p_text and len(p_text) > 20:
                sections.append(p_text)
    except Exception as exc:
        logger.debug(f"XML parse error in fulltext extraction: {exc}")
    return sections
