"""Tests for pharos.tools.pubmed_tools — PubMed search and abstract parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pharos.config import Settings
from pharos.tools.pubmed_tools import PubMedArticle, PubMedClient, _parse_pubmed_xml

# ------------------------------------------------------------------
# Sample XML fixtures
# ------------------------------------------------------------------

SAMPLE_EFETCH_XML = """\
<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN"
 "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">12345678</PMID>
      <Article PubModel="Print">
        <Journal>
          <Title>Nature Medicine</Title>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>KRAS G12C inhibitors in lung cancer</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">KRAS mutations are common in NSCLC.</AbstractText>
          <AbstractText Label="RESULTS">We found that sotorasib inhibits KRAS G12C.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
          </Author>
        </AuthorList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName>Lung Neoplasms</DescriptorName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName>Proto-Oncogene Proteins p21(ras)</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1038/s41591-024-00001-x</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">87654321</PMID>
      <Article PubModel="Print">
        <Journal>
          <Title>Cell</Title>
          <JournalIssue>
            <PubDate>
              <Year>2023</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>EGFR signaling review</ArticleTitle>
        <Abstract>
          <AbstractText>EGFR drives tumor growth.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Lee</LastName>
            <ForeName>Alice</ForeName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">87654321</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


# ------------------------------------------------------------------
# XML parsing tests
# ------------------------------------------------------------------


class TestParsePubmedXml:
    """Test the XML parser on sample PubMed responses."""

    def test_parses_two_articles(self) -> None:
        articles = _parse_pubmed_xml(SAMPLE_EFETCH_XML)
        assert len(articles) == 2

    def test_first_article_fields(self) -> None:
        articles = _parse_pubmed_xml(SAMPLE_EFETCH_XML)
        a = articles[0]
        assert a.pmid == "12345678"
        assert a.title == "KRAS G12C inhibitors in lung cancer"
        assert "KRAS mutations" in a.abstract
        assert "sotorasib" in a.abstract
        assert a.authors == ["Smith John", "Doe Jane"]
        assert a.journal == "Nature Medicine"
        assert a.year == 2024
        assert "Lung Neoplasms" in a.mesh_terms
        assert a.doi == "10.1038/s41591-024-00001-x"

    def test_structured_abstract_labels(self) -> None:
        articles = _parse_pubmed_xml(SAMPLE_EFETCH_XML)
        abstract = articles[0].abstract
        assert "BACKGROUND:" in abstract
        assert "RESULTS:" in abstract

    def test_second_article_no_doi(self) -> None:
        articles = _parse_pubmed_xml(SAMPLE_EFETCH_XML)
        a = articles[1]
        assert a.pmid == "87654321"
        assert a.doi is None
        assert a.mesh_terms == []

    def test_empty_xml(self) -> None:
        xml = '<?xml version="1.0" ?><PubmedArticleSet></PubmedArticleSet>'
        articles = _parse_pubmed_xml(xml)
        assert articles == []


# ------------------------------------------------------------------
# PubMedClient tests with mocked httpx
# ------------------------------------------------------------------


@pytest.fixture
def pubmed_client() -> PubMedClient:
    """Create a PubMedClient with fast settings for tests."""
    settings = Settings()
    settings.max_retries = 1
    settings.retry_base_delay = 0.01
    settings.request_timeout = 5.0
    return PubMedClient(settings)


def _mock_response(
    *, status_code: int = 200, json_data: dict | None = None, text: str = ""
) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    resp.request = MagicMock()
    return resp


class TestPubMedClientSearch:
    """Test the search method."""

    async def test_search_returns_pmids(self, pubmed_client: PubMedClient) -> None:
        resp = _mock_response(
            json_data={
                "esearchresult": {
                    "count": "3",
                    "idlist": ["111", "222", "333"],
                }
            }
        )
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("pharos.tools.pubmed_tools.httpx.AsyncClient", return_value=mock_http):
            pmids = await pubmed_client.search("KRAS lung cancer")

        assert pmids == ["111", "222", "333"]

    async def test_search_with_date_filters(self, pubmed_client: PubMedClient) -> None:
        resp = _mock_response(json_data={"esearchresult": {"idlist": ["111"]}})
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("pharos.tools.pubmed_tools.httpx.AsyncClient", return_value=mock_http):
            pmids = await pubmed_client.search(
                "TP53", min_date="2023/01/01", max_date="2024/12/31"
            )

        assert pmids == ["111"]
        call_kwargs = mock_http.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert params.get("mindate") == "2023/01/01"
        assert params.get("maxdate") == "2024/12/31"

    async def test_search_empty_results(self, pubmed_client: PubMedClient) -> None:
        resp = _mock_response(json_data={"esearchresult": {"idlist": []}})
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("pharos.tools.pubmed_tools.httpx.AsyncClient", return_value=mock_http):
            pmids = await pubmed_client.search("nonexistent_query_xyz")

        assert pmids == []


class TestPubMedClientFetch:
    """Test the fetch_abstracts method."""

    async def test_fetch_abstracts_parses_xml(self, pubmed_client: PubMedClient) -> None:
        resp = _mock_response(text=SAMPLE_EFETCH_XML)
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("pharos.tools.pubmed_tools.httpx.AsyncClient", return_value=mock_http):
            articles = await pubmed_client.fetch_abstracts(["12345678", "87654321"])

        assert len(articles) == 2
        assert articles[0].pmid == "12345678"

    async def test_fetch_empty_list(self, pubmed_client: PubMedClient) -> None:
        articles = await pubmed_client.fetch_abstracts([])
        assert articles == []

    async def test_fetch_full_metadata(self, pubmed_client: PubMedClient) -> None:
        resp = _mock_response(text=SAMPLE_EFETCH_XML)
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("pharos.tools.pubmed_tools.httpx.AsyncClient", return_value=mock_http):
            article = await pubmed_client.fetch_full_metadata("12345678")

        assert article.title == "KRAS G12C inhibitors in lung cancer"
        assert article.doi == "10.1038/s41591-024-00001-x"


class TestPubMedClientRetry:
    """Test retry behaviour on transient errors."""

    async def test_retries_on_429(self, pubmed_client: PubMedClient) -> None:
        rate_limit_resp = MagicMock(spec=httpx.Response)
        rate_limit_resp.status_code = 429
        rate_limit_resp.request = MagicMock()

        ok_resp = _mock_response(json_data={"esearchresult": {"idlist": ["999"]}})

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=[rate_limit_resp, ok_resp])
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("pharos.tools.pubmed_tools.httpx.AsyncClient", return_value=mock_http):
            pmids = await pubmed_client.search("test")

        assert pmids == ["999"]
        assert mock_http.get.call_count == 2

    async def test_retries_on_connection_error(self, pubmed_client: PubMedClient) -> None:
        ok_resp = _mock_response(json_data={"esearchresult": {"idlist": ["123"]}})
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=[httpx.ConnectError("fail"), ok_resp])
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("pharos.tools.pubmed_tools.httpx.AsyncClient", return_value=mock_http):
            pmids = await pubmed_client.search("test")

        assert pmids == ["123"]


class TestPubMedArticleModel:
    """Test the PubMedArticle pydantic model."""

    def test_minimal_article(self) -> None:
        a = PubMedArticle(pmid="123")
        assert a.pmid == "123"
        assert a.title == ""
        assert a.doi is None
        assert a.mesh_terms == []

    def test_full_article(self) -> None:
        a = PubMedArticle(
            pmid="456",
            title="Test",
            abstract="Abstract text",
            authors=["Author One"],
            journal="Nature",
            year=2024,
            mesh_terms=["Term1"],
            doi="10.1000/test",
        )
        assert a.year == 2024
        assert len(a.authors) == 1
