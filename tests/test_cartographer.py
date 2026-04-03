"""Tests for pharos.agents.cartographer — KG building pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.agents.cartographer import CartographerAgent, _deduplicate_entities, _map_entity_type
from pharos.config import Settings
from pharos.graph.entity_extractor import BioEntity, BioEntityExtractor, BioRelation
from pharos.orchestration.task_models import Task, TaskType, WorkflowState
from pharos.tools.pubmed_tools import PubMedArticle, PubMedClient


@pytest.fixture
def mock_pubmed() -> MagicMock:
    """Mock PubMedClient."""
    client = MagicMock(spec=PubMedClient)
    client.search = AsyncMock(return_value=["111", "222"])
    client.fetch_abstracts = AsyncMock(
        return_value=[
            PubMedArticle(
                pmid="111",
                title="KRAS in NSCLC",
                abstract="KRAS G12C mutations drive non-small cell lung cancer progression.",
            ),
            PubMedArticle(
                pmid="222",
                title="Sotorasib efficacy",
                abstract="Sotorasib inhibits KRAS G12C and shows efficacy in lung cancer.",
            ),
        ]
    )
    return client


@pytest.fixture
def mock_extractor() -> MagicMock:
    """Mock BioEntityExtractor."""
    extractor = MagicMock(spec=BioEntityExtractor)
    extractor.extract_entities = AsyncMock(
        side_effect=[
            [
                BioEntity(name="kras", entity_type="Gene"),
                BioEntity(name="non-small cell lung cancer", entity_type="Disease"),
            ],
            [
                BioEntity(name="sotorasib", entity_type="Drug"),
                BioEntity(name="kras", entity_type="Gene", aliases=["k-ras"]),
                BioEntity(name="lung cancer", entity_type="Disease"),
            ],
        ]
    )
    extractor.extract_relations = AsyncMock(
        side_effect=[
            [
                BioRelation(
                    source="kras",
                    target="non-small cell lung cancer",
                    relation_type="ASSOCIATED_WITH",
                    confidence=0.9,
                    evidence="KRAS G12C mutations drive NSCLC",
                ),
            ],
            [
                BioRelation(
                    source="sotorasib",
                    target="kras",
                    relation_type="INHIBITS",
                    confidence=0.85,
                    evidence="Sotorasib inhibits KRAS G12C",
                ),
            ],
        ]
    )
    return extractor


@pytest.fixture
def cartographer(
    mock_ollama: MagicMock,
    mock_neo4j: MagicMock,
    mock_pubmed: MagicMock,
    mock_extractor: MagicMock,
) -> CartographerAgent:
    """Create a CartographerAgent with all mocked dependencies."""
    return CartographerAgent(
        ollama=mock_ollama,
        kg=mock_neo4j,
        config=Settings(),
        pubmed=mock_pubmed,
        extractor=mock_extractor,
    )


@pytest.fixture
def state() -> WorkflowState:
    """Create an initial workflow state for Cartographer."""
    return WorkflowState(
        task=Task(query="test", task_type=TaskType.BUILD_KG),
        results=[],
        current_agent="cartographer",
        kg_context="",
        iteration=0,
    )


class TestCartographerRun:
    """Test the full Cartographer pipeline."""

    async def test_builds_kg_from_pubmed(
        self,
        cartographer: CartographerAgent,
        state: WorkflowState,
    ) -> None:
        task = Task(query="Build a knowledge graph about KRAS mutations in lung cancer")
        result = await cartographer.run(task, state)

        assert result.agent_name == "Cartographer"
        assert result.artifacts["nodes_added"] > 0
        assert result.artifacts["relations_added"] > 0
        assert result.artifacts["articles_processed"] == 2
        assert "kras" in result.artifacts["top_entities"]

    async def test_calls_pubmed_search(
        self,
        cartographer: CartographerAgent,
        state: WorkflowState,
    ) -> None:
        task = Task(query="Map EGFR interactions")
        await cartographer.run(task, state)

        cartographer.pubmed.search.assert_called_once_with("Map EGFR interactions", max_results=20)

    async def test_calls_fetch_abstracts(
        self,
        cartographer: CartographerAgent,
        state: WorkflowState,
    ) -> None:
        task = Task(query="test query")
        await cartographer.run(task, state)

        cartographer.pubmed.fetch_abstracts.assert_called_once_with(["111", "222"])

    async def test_adds_nodes_to_kg(
        self,
        cartographer: CartographerAgent,
        mock_neo4j: MagicMock,
        state: WorkflowState,
    ) -> None:
        task = Task(query="test query")
        await cartographer.run(task, state)

        assert mock_neo4j.add_node.call_count > 0

    async def test_adds_relations_to_kg(
        self,
        cartographer: CartographerAgent,
        mock_neo4j: MagicMock,
        state: WorkflowState,
    ) -> None:
        task = Task(query="test query")
        result = await cartographer.run(task, state)

        assert mock_neo4j.add_relation.call_count > 0
        assert len(result.kg_updates) > 0

    async def test_handles_no_pubmed_results(
        self,
        cartographer: CartographerAgent,
        state: WorkflowState,
    ) -> None:
        cartographer.pubmed.search = AsyncMock(return_value=[])
        task = Task(query="nonexistent topic xyz")
        result = await cartographer.run(task, state)

        assert "No PubMed articles found" in result.content
        assert result.confidence < 0.5

    async def test_confidence_scales_with_articles(
        self,
        cartographer: CartographerAgent,
        state: WorkflowState,
    ) -> None:
        task = Task(query="test query")
        result = await cartographer.run(task, state)

        # 2 articles -> 0.3 + 0.03*2 = 0.36
        assert result.confidence > 0.3


# ------------------------------------------------------------------
# Helper function tests
# ------------------------------------------------------------------


class TestDeduplicateEntities:
    """Test entity deduplication."""

    def test_deduplicates_by_name_and_type(self) -> None:
        entities = [
            BioEntity(name="kras", entity_type="Gene"),
            BioEntity(name="kras", entity_type="Gene", aliases=["k-ras"]),
        ]
        result = _deduplicate_entities(entities)
        assert len(result) == 1
        assert "k-ras" in result[0].aliases

    def test_keeps_different_types(self) -> None:
        entities = [
            BioEntity(name="egfr", entity_type="Gene"),
            BioEntity(name="egfr", entity_type="Protein"),
        ]
        result = _deduplicate_entities(entities)
        assert len(result) == 2

    def test_merges_identifiers(self) -> None:
        entities = [
            BioEntity(name="tp53", entity_type="Gene", identifiers={"ncbi": "7157"}),
            BioEntity(name="tp53", entity_type="Gene", identifiers={"ensembl": "ENSG001"}),
        ]
        result = _deduplicate_entities(entities)
        assert len(result) == 1
        assert result[0].identifiers == {"ncbi": "7157", "ensembl": "ENSG001"}

    def test_empty_list(self) -> None:
        assert _deduplicate_entities([]) == []


class TestMapEntityType:
    """Test entity type to node type mapping."""

    def test_maps_known_types(self) -> None:
        assert _map_entity_type("Gene") == "Gene"
        assert _map_entity_type("drug") == "Drug"
        assert _map_entity_type("CellType") == "Phenotype"
        assert _map_entity_type("Organism") == "Anatomy"

    def test_unknown_type_returns_entity(self) -> None:
        assert _map_entity_type("Unknown") == "Entity"
