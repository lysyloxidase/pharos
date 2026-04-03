"""Tests for pharos.agents.scribe — narrative review writer pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.agents.scribe import (
    DraftSection,
    Reference,
    ReviewDraft,
    ReviewSection,
    ScribeAgent,
    _assemble_markdown,
    _build_references,
    _format_apa_reference,
    _parse_critique,
    _parse_outline,
    _parse_search_queries,
)
from pharos.config import Settings
from pharos.graph.entity_extractor import BioEntityExtractor
from pharos.orchestration.task_models import Task, TaskType, WorkflowState
from pharos.tools.pubmed_tools import PubMedArticle, PubMedClient

# ------------------------------------------------------------------
# Sample data
# ------------------------------------------------------------------

SAMPLE_OUTLINE_JSON = json.dumps(
    {
        "title": "KRAS Mutations in Non-Small Cell Lung Cancer: Therapeutic Advances",
        "sections": [
            {
                "heading": "Introduction",
                "key_questions": [
                    "What is the prevalence of KRAS mutations in NSCLC?",
                    "Why has KRAS been historically undruggable?",
                ],
            },
            {
                "heading": "KRAS G12C Inhibitors",
                "key_questions": [
                    "How do covalent KRAS G12C inhibitors work?",
                    "What are the clinical outcomes of sotorasib and adagrasib?",
                ],
            },
            {
                "heading": "Resistance Mechanisms",
                "key_questions": [
                    "What acquired resistance mechanisms have been identified?",
                ],
            },
            {
                "heading": "Future Directions",
                "key_questions": [
                    "What combination strategies are being explored?",
                ],
            },
        ],
    }
)

SAMPLE_SEARCH_QUERIES_JSON = json.dumps(
    [
        "KRAS G12C inhibitors NSCLC",
        "sotorasib clinical trial lung cancer",
    ]
)


def _make_article(pmid: str, author: str, year: int, title: str, abstract: str) -> PubMedArticle:
    return PubMedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=[f"{author} A"],
        journal="Nature Medicine",
        year=year,
        doi=f"10.1000/test.{pmid}",
    )


SAMPLE_ARTICLES = [
    _make_article(
        "111", "Smith", 2023, "Sotorasib in NSCLC", "Sotorasib showed 37% ORR in KRAS G12C NSCLC."
    ),
    _make_article(
        "222",
        "Jones",
        2024,
        "Adagrasib trial results",
        "Adagrasib demonstrated durable responses.",
    ),
    _make_article(
        "333",
        "Lee",
        2023,
        "KRAS resistance",
        "Resistance arises via KRAS amplification and bypass pathways.",
    ),
]

SAMPLE_CRITIQUE_GOOD = json.dumps(
    {
        "score": 8,
        "issues": [],
        "has_unsupported_claims": False,
        "has_hallucinations": False,
        "is_coherent": True,
    }
)

SAMPLE_CRITIQUE_BAD = json.dumps(
    {
        "score": 4,
        "issues": ["Missing citations for prevalence claim", "Logical gap in paragraph 2"],
        "has_unsupported_claims": True,
        "has_hallucinations": False,
        "is_coherent": False,
    }
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_pubmed() -> MagicMock:
    """Mock PubMedClient returning sample articles."""
    client = MagicMock(spec=PubMedClient)
    client.search = AsyncMock(return_value=["111", "222", "333"])
    client.fetch_abstracts = AsyncMock(return_value=SAMPLE_ARTICLES)
    return client


@pytest.fixture
def mock_extractor() -> MagicMock:
    """Mock BioEntityExtractor."""
    ext = MagicMock(spec=BioEntityExtractor)
    ext.extract_entities = AsyncMock(return_value=[])
    ext.extract_relations = AsyncMock(return_value=[])
    return ext


@pytest.fixture
def scribe(
    mock_ollama: MagicMock,
    mock_neo4j: MagicMock,
    mock_pubmed: MagicMock,
    mock_extractor: MagicMock,
) -> ScribeAgent:
    """Create a ScribeAgent with all mocked dependencies."""
    return ScribeAgent(
        ollama=mock_ollama,
        kg=mock_neo4j,
        config=Settings(),
        pubmed=mock_pubmed,
        extractor=mock_extractor,
    )


@pytest.fixture
def state() -> WorkflowState:
    """Sample workflow state."""
    return WorkflowState(
        task=Task(query="test", task_type=TaskType.REVIEW),
        results=[],
        current_agent="scribe",
        kg_context="",
        iteration=0,
    )


# ------------------------------------------------------------------
# Parser tests
# ------------------------------------------------------------------


class TestParseOutline:
    """Test outline JSON parsing."""

    def test_parses_valid_outline(self) -> None:
        outline = _parse_outline(SAMPLE_OUTLINE_JSON)
        assert (
            outline.title == "KRAS Mutations in Non-Small Cell Lung Cancer: Therapeutic Advances"
        )
        assert len(outline.sections) == 4
        assert outline.sections[0].heading == "Introduction"
        assert len(outline.sections[0].key_questions) == 2

    def test_handles_invalid_json(self) -> None:
        outline = _parse_outline("not json at all")
        assert outline.title == "Untitled Review"
        assert outline.sections == []

    def test_handles_markdown_wrapped(self) -> None:
        wrapped = f"```json\n{SAMPLE_OUTLINE_JSON}\n```"
        outline = _parse_outline(wrapped)
        assert len(outline.sections) == 4

    def test_skips_sections_without_heading(self) -> None:
        data = json.dumps(
            {
                "title": "Test",
                "sections": [
                    {"heading": "Good", "key_questions": []},
                    {"key_questions": ["no heading"]},
                ],
            }
        )
        outline = _parse_outline(data)
        assert len(outline.sections) == 1


class TestParseSearchQueries:
    """Test search query JSON parsing."""

    def test_parses_array(self) -> None:
        queries = _parse_search_queries('["query1", "query2"]')
        assert queries == ["query1", "query2"]

    def test_parses_dict_wrapped(self) -> None:
        queries = _parse_search_queries('{"queries": ["q1", "q2"]}')
        assert queries == ["q1", "q2"]

    def test_returns_empty_on_bad_json(self) -> None:
        assert _parse_search_queries("broken") == []


class TestParseCritique:
    """Test critique JSON parsing."""

    def test_parses_good_critique(self) -> None:
        score, issues = _parse_critique(SAMPLE_CRITIQUE_GOOD)
        assert score == 8
        assert issues == []

    def test_parses_bad_critique(self) -> None:
        score, issues = _parse_critique(SAMPLE_CRITIQUE_BAD)
        assert score == 4
        assert len(issues) == 2

    def test_defaults_on_bad_json(self) -> None:
        score, issues = _parse_critique("not json")
        assert score == 5
        assert len(issues) == 1

    def test_clamps_score(self) -> None:
        score, _ = _parse_critique(json.dumps({"score": 99}))
        assert score == 10
        score2, _ = _parse_critique(json.dumps({"score": -5}))
        assert score2 == 1


# ------------------------------------------------------------------
# Reference building tests
# ------------------------------------------------------------------


class TestBuildReferences:
    """Test citation key generation."""

    def test_builds_keys_from_first_author(self) -> None:
        refs = _build_references(SAMPLE_ARTICLES)
        assert refs[0].key == "Smith2023"
        assert refs[1].key == "Jones2024"

    def test_deduplicates_keys(self) -> None:
        arts = [
            _make_article("1", "Smith", 2024, "A", "a"),
            _make_article("2", "Smith", 2024, "B", "b"),
            _make_article("3", "Smith", 2024, "C", "c"),
        ]
        refs = _build_references(arts)
        keys = [r.key for r in refs]
        assert keys == ["Smith2024", "Smith2024a", "Smith2024b"]

    def test_handles_no_authors(self) -> None:
        art = PubMedArticle(pmid="999", year=2024)
        refs = _build_references([art])
        assert refs[0].key == "Unknown2024"

    def test_preserves_pmid_and_doi(self) -> None:
        refs = _build_references(SAMPLE_ARTICLES[:1])
        assert refs[0].pmid == "111"
        assert refs[0].doi == "10.1000/test.111"


class TestFormatApaReference:
    """Test APA formatting."""

    def test_formats_with_doi(self) -> None:
        ref = Reference(
            key="Smith2024",
            pmid="123",
            authors=["Smith A", "Jones B"],
            title="Test Article",
            journal="Nature",
            year=2024,
            doi="10.1000/test",
        )
        apa = _format_apa_reference(ref)
        assert "Smith A, Jones B" in apa
        assert "(2024)" in apa
        assert "Test Article" in apa
        assert "*Nature*" in apa
        assert "https://doi.org/10.1000/test" in apa
        assert "PMID: 123" in apa

    def test_formats_without_doi(self) -> None:
        ref = Reference(key="X2024", pmid="456", title="T", journal="J", year=2024)
        apa = _format_apa_reference(ref)
        assert "doi.org" not in apa


class TestAssembleMarkdown:
    """Test final Markdown assembly."""

    def test_includes_all_parts(self) -> None:
        draft = ReviewDraft(
            title="My Review",
            abstract="This is the abstract.",
            sections=[DraftSection(heading="Intro", body="Intro text.", score=8)],
            references=[
                Reference(
                    key="Smith2024",
                    pmid="1",
                    authors=["Smith A"],
                    title="T",
                    journal="J",
                    year=2024,
                ),
            ],
        )
        md = _assemble_markdown(draft)
        assert "# My Review" in md
        assert "## Abstract" in md
        assert "This is the abstract." in md
        assert "## Intro" in md
        assert "Intro text." in md
        assert "## References" in md
        assert "[Smith2024]" in md

    def test_handles_no_abstract(self) -> None:
        draft = ReviewDraft(title="T", sections=[], references=[])
        md = _assemble_markdown(draft)
        assert "# T" in md
        assert "Abstract" not in md


# ------------------------------------------------------------------
# Scribe agent pipeline tests
# ------------------------------------------------------------------


class TestScribeOutlineGeneration:
    """Test outline generation step."""

    async def test_generates_outline(self, scribe: ScribeAgent) -> None:
        scribe.ollama.generate = AsyncMock(return_value=SAMPLE_OUTLINE_JSON)
        outline = await scribe._generate_outline("KRAS in NSCLC")
        assert outline.title != ""
        assert len(outline.sections) == 4

    async def test_handles_outline_failure(self, scribe: ScribeAgent) -> None:
        scribe.ollama.generate = AsyncMock(return_value="garbage")
        outline = await scribe._generate_outline("test")
        assert outline.sections == []


class TestScribeSectionDrafting:
    """Test section drafting with critique loop."""

    async def test_drafts_section_without_rewrite(self, scribe: ScribeAgent) -> None:
        scribe.ollama.generate = AsyncMock(
            side_effect=[
                "This is a well-written section [Smith2023].",
                SAMPLE_CRITIQUE_GOOD,
            ]
        )
        section = ReviewSection(heading="Test", key_questions=["Q1?"])
        refs = _build_references(SAMPLE_ARTICLES[:1])

        drafted = await scribe._draft_section(section, SAMPLE_ARTICLES[:1], refs, "")
        assert drafted.score == 8
        assert drafted.body != ""

    async def test_rewrites_on_low_score(self, scribe: ScribeAgent) -> None:
        scribe.ollama.generate = AsyncMock(
            side_effect=[
                "Bad draft without citations.",
                SAMPLE_CRITIQUE_BAD,
                "Improved draft [Smith2023] with citations.",
                SAMPLE_CRITIQUE_GOOD,
            ]
        )
        section = ReviewSection(heading="Test", key_questions=["Q1?"])
        refs = _build_references(SAMPLE_ARTICLES[:1])

        drafted = await scribe._draft_section(section, SAMPLE_ARTICLES[:1], refs, "")
        assert drafted.score == 8
        # generate called 4 times: initial draft, critique, rewrite, critique
        assert scribe.ollama.generate.call_count == 4

    async def test_stops_after_max_iterations(self, scribe: ScribeAgent) -> None:
        scribe.ollama.generate = AsyncMock(
            side_effect=[
                "Draft v1",
                SAMPLE_CRITIQUE_BAD,
                "Draft v2",
                SAMPLE_CRITIQUE_BAD,
                "Draft v3",
                SAMPLE_CRITIQUE_BAD,
            ]
        )
        section = ReviewSection(heading="Test", key_questions=["Q1?"])
        refs = _build_references(SAMPLE_ARTICLES[:1])

        drafted = await scribe._draft_section(section, SAMPLE_ARTICLES[:1], refs, "")
        # Score remains low but we stop at max iterations
        assert drafted.score == 4
        # 3 iterations: draft+critique, rewrite+critique, rewrite+critique
        assert scribe.ollama.generate.call_count == 6


class TestScribeFullReview:
    """Integration test of the full Scribe pipeline."""

    async def test_full_review_pipeline(
        self,
        scribe: ScribeAgent,
        state: WorkflowState,
    ) -> None:
        # Mock LLM responses in order:
        # 1. outline, 2-5. search queries (4 sections), 6+. drafts and critiques
        scribe.ollama.generate = AsyncMock(
            side_effect=[
                # Outline
                SAMPLE_OUTLINE_JSON,
                # Search queries for 4 sections
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                # Section 1: draft + critique
                "Introduction text [Smith2023].",
                SAMPLE_CRITIQUE_GOOD,
                # Section 2: draft + critique
                "KRAS G12C inhibitors [Jones2024].",
                SAMPLE_CRITIQUE_GOOD,
                # Section 3: draft + critique
                "Resistance mechanisms [Lee2023].",
                SAMPLE_CRITIQUE_GOOD,
                # Section 4: draft + critique
                "Future directions [Smith2023].",
                SAMPLE_CRITIQUE_GOOD,
                # Abstract
                "This review covers KRAS therapeutic advances.",
            ]
        )

        task = Task(query="Write a review of KRAS mutations in NSCLC")
        result = await scribe.run(task, state)

        assert result.agent_name == "Scribe"
        assert "# KRAS Mutations" in result.content
        assert "## Abstract" in result.content
        assert "## References" in result.content
        assert result.artifacts["reference_count"] > 0
        assert len(result.artifacts["sections"]) == 4
        assert result.confidence > 0

    async def test_returns_early_on_empty_outline(
        self,
        scribe: ScribeAgent,
        state: WorkflowState,
    ) -> None:
        scribe.ollama.generate = AsyncMock(return_value="invalid json")
        task = Task(query="bad topic")
        result = await scribe.run(task, state)

        assert "Failed to generate" in result.content
        assert result.confidence < 0.5

    async def test_queries_kg_for_context(
        self,
        scribe: ScribeAgent,
        mock_neo4j: MagicMock,
        state: WorkflowState,
    ) -> None:
        mock_neo4j.search_nodes = AsyncMock(
            return_value=[{"n": {"name": "KRAS", "_labels": ["Gene"]}}]
        )
        scribe.ollama.generate = AsyncMock(
            side_effect=[
                SAMPLE_OUTLINE_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                "Intro [Smith2023].",
                SAMPLE_CRITIQUE_GOOD,
                "Body [Jones2024].",
                SAMPLE_CRITIQUE_GOOD,
                "Resist [Lee2023].",
                SAMPLE_CRITIQUE_GOOD,
                "Future [Smith2023].",
                SAMPLE_CRITIQUE_GOOD,
                "Abstract text.",
            ]
        )
        task = Task(query="KRAS review")
        await scribe.run(task, state)

        mock_neo4j.search_nodes.assert_called_once()

    async def test_kg_context_failure_is_graceful(
        self,
        scribe: ScribeAgent,
        mock_neo4j: MagicMock,
        state: WorkflowState,
    ) -> None:
        mock_neo4j.search_nodes = AsyncMock(side_effect=Exception("KG down"))
        scribe.ollama.generate = AsyncMock(
            side_effect=[
                SAMPLE_OUTLINE_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                SAMPLE_SEARCH_QUERIES_JSON,
                "Intro [Smith2023].",
                SAMPLE_CRITIQUE_GOOD,
                "Body [Jones2024].",
                SAMPLE_CRITIQUE_GOOD,
                "Resist [Lee2023].",
                SAMPLE_CRITIQUE_GOOD,
                "Future [Smith2023].",
                SAMPLE_CRITIQUE_GOOD,
                "Abstract text.",
            ]
        )
        task = Task(query="KRAS review")
        result = await scribe.run(task, state)

        # Should still produce a review despite KG failure
        assert "# KRAS Mutations" in result.content
