"""Async client for NCBI E-utilities API — PubMed search and abstract retrieval."""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

if TYPE_CHECKING:
    from pharos.config import Settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 200


class PubMedArticle(BaseModel):
    """Structured representation of a PubMed article.

    Attributes:
        pmid: PubMed identifier.
        title: Article title.
        abstract: Abstract text.
        authors: List of author names ("Last First" format).
        journal: Journal title.
        year: Publication year.
        mesh_terms: Assigned MeSH descriptor terms.
        doi: Digital Object Identifier (if available).
    """

    pmid: str
    title: str = ""
    abstract: str = ""
    authors: list[str] = []
    journal: str = ""
    year: int = 0
    mesh_terms: list[str] = []
    doi: str | None = None


class PubMedClient:
    """Async client for NCBI E-utilities with rate limiting and retry.

    Args:
        config: Application settings (provides API key and rate limit).
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, config: Settings | None = None) -> None:
        from pharos.config import get_settings

        cfg = config or get_settings()
        self._api_key = cfg.ncbi_api_key
        self._rate_limit = 10.0 if self._api_key else cfg.pubmed_rate_limit
        self._max_retries = cfg.max_retries
        self._base_delay = cfg.retry_base_delay
        self._timeout = cfg.request_timeout
        self._min_interval = 1.0 / self._rate_limit
        self._last_request_time = 0.0

    def _base_params(self) -> dict[str, str]:
        """Return common query parameters for all E-utility requests."""
        params: dict[str, str] = {}
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def _throttle(self) -> None:
        """Enforce rate limit by sleeping if requests are too frequent."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _request(self, url: str, params: dict[str, Any]) -> httpx.Response:
        """Send a GET request with retry and rate limiting.

        Args:
            url: Full E-utility endpoint URL.
            params: Query parameters.

        Returns:
            httpx Response.

        Raises:
            httpx.HTTPStatusError: If all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            await self._throttle()
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(url, params=params)
                    if response.status_code in (429, 503):
                        raise httpx.HTTPStatusError(
                            f"Rate limited ({response.status_code})",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    return response
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._base_delay * (2**attempt)
                    logger.warning(
                        "PubMed request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int = 50,
        min_date: str | None = None,
        max_date: str | None = None,
    ) -> list[str]:
        """Search PubMed and return matching PMIDs.

        Args:
            query: PubMed search query.
            max_results: Maximum number of PMIDs to return.
            min_date: Optional minimum date filter (YYYY/MM/DD).
            max_date: Optional maximum date filter (YYYY/MM/DD).

        Returns:
            List of PMID strings.
        """
        params: dict[str, Any] = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(max_results),
        }
        if min_date:
            params["mindate"] = min_date
            params["datetype"] = "pdat"
        if max_date:
            params["maxdate"] = max_date
            if "datetype" not in params:
                params["datetype"] = "pdat"

        url = f"{self.BASE_URL}/esearch.fcgi"
        response = await self._request(url, params)
        data: dict[str, Any] = response.json()
        result = data.get("esearchresult", {})
        pmids: list[str] = result.get("idlist", [])
        logger.info("PubMed search '%s' returned %d PMIDs", query, len(pmids))
        return pmids

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch article metadata and abstracts for a list of PMIDs.

        Retrieves in batches of 200 to respect NCBI limits.

        Args:
            pmids: List of PubMed IDs to fetch.

        Returns:
            List of PubMedArticle objects.
        """
        if not pmids:
            return []

        articles: list[PubMedArticle] = []
        for i in range(0, len(pmids), _BATCH_SIZE):
            batch = pmids[i : i + _BATCH_SIZE]
            batch_articles = await self._fetch_batch(batch)
            articles.extend(batch_articles)

        return articles

    async def fetch_full_metadata(self, pmid: str) -> PubMedArticle:
        """Fetch complete metadata for a single PMID.

        Args:
            pmid: PubMed identifier.

        Returns:
            PubMedArticle with all available fields populated.
        """
        results = await self._fetch_batch([pmid])
        if not results:
            return PubMedArticle(pmid=pmid)
        return results[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_batch(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch a single batch of PMIDs via efetch.

        Args:
            pmids: Batch of PMIDs (max 200).

        Returns:
            Parsed list of PubMedArticle objects.
        """
        params: dict[str, Any] = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        url = f"{self.BASE_URL}/efetch.fcgi"
        response = await self._request(url, params)
        return _parse_pubmed_xml(response.text)


def _parse_pubmed_xml(xml_text: str) -> list[PubMedArticle]:
    """Parse PubMed efetch XML into PubMedArticle objects.

    Args:
        xml_text: Raw XML response from efetch.

    Returns:
        List of parsed articles.
    """
    articles: list[PubMedArticle] = []
    root = ET.fromstring(xml_text)

    for article_el in root.iter("PubmedArticle"):
        articles.append(_parse_article(article_el))

    return articles


def _parse_article(article_el: ET.Element) -> PubMedArticle:
    """Parse a single <PubmedArticle> element.

    Args:
        article_el: XML element for one PubMed article.

    Returns:
        Populated PubMedArticle.
    """
    # PMID
    pmid_el = article_el.find(".//PMID")
    pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""

    # Title
    title_el = article_el.find(".//ArticleTitle")
    title = _text_content(title_el) if title_el is not None else ""

    # Abstract — may have multiple AbstractText sections
    abstract_parts: list[str] = []
    for abs_el in article_el.findall(".//AbstractText"):
        label = abs_el.get("Label", "")
        text = _text_content(abs_el)
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # Authors
    authors: list[str] = []
    for author_el in article_el.findall(".//Author"):
        last = author_el.findtext("LastName", "")
        first = author_el.findtext("ForeName", "")
        if last:
            authors.append(f"{last} {first}".strip())

    # Journal
    journal = article_el.findtext(".//Journal/Title", "")

    # Year
    year_text = article_el.findtext(".//PubDate/Year", "")
    if not year_text:
        medline_date = article_el.findtext(".//PubDate/MedlineDate", "")
        year_text = medline_date[:4] if len(medline_date) >= 4 else "0"
    try:
        year = int(year_text)
    except ValueError:
        year = 0

    # MeSH terms
    mesh_terms: list[str] = []
    for mesh_el in article_el.findall(".//MeshHeading/DescriptorName"):
        mesh_text = mesh_el.text
        if mesh_text:
            mesh_terms.append(mesh_text)

    # DOI
    doi: str | None = None
    for id_el in article_el.findall(".//ArticleId"):
        if id_el.get("IdType") == "doi":
            doi = id_el.text
            break

    return PubMedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        year=year,
        mesh_terms=mesh_terms,
        doi=doi,
    )


def _text_content(el: ET.Element) -> str:
    """Extract all text from an element, including mixed content with sub-tags.

    Args:
        el: XML element.

    Returns:
        Concatenated text content.
    """
    parts: list[str] = []
    if el.text:
        parts.append(el.text)
    for child in el:
        if child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    if el.tail:
        pass  # tail belongs to parent, not this element
    return "".join(parts)
