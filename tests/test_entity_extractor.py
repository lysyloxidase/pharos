"""Tests for pharos.graph.entity_extractor — entity and relation extraction."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.config import Settings
from pharos.graph.entity_extractor import (
    BioEntity,
    BioEntityExtractor,
    _extract_json,
    _normalize_name,
)


@pytest.fixture
def extractor(mock_ollama: MagicMock) -> BioEntityExtractor:
    """Create a BioEntityExtractor with mocked Ollama."""
    return BioEntityExtractor(mock_ollama, Settings())


# ------------------------------------------------------------------
# _extract_json tests
# ------------------------------------------------------------------


class TestExtractJson:
    """Test JSON extraction from various LLM output formats."""

    def test_plain_json_array(self) -> None:
        raw = '[{"name": "TP53", "entity_type": "Gene"}]'
        result = _extract_json(raw)
        assert isinstance(result, list)
        assert result[0]["name"] == "TP53"

    def test_markdown_code_fence(self) -> None:
        raw = '```json\n[{"name": "EGFR", "entity_type": "Protein"}]\n```'
        result = _extract_json(raw)
        assert isinstance(result, list)
        assert result[0]["name"] == "EGFR"

    def test_json_with_surrounding_text(self) -> None:
        raw = 'Here are the entities:\n[{"name": "aspirin", "entity_type": "Drug"}]\nDone.'
        result = _extract_json(raw)
        assert isinstance(result, list)
        assert result[0]["name"] == "aspirin"

    def test_empty_string(self) -> None:
        assert _extract_json("") is None

    def test_invalid_json(self) -> None:
        assert _extract_json("not json at all") is None

    def test_json_object(self) -> None:
        raw = '{"entities": [{"name": "BRCA1", "entity_type": "Gene"}]}'
        result = _extract_json(raw)
        assert isinstance(result, dict)


class TestNormalizeName:
    """Test entity name normalization."""

    def test_strips_whitespace(self) -> None:
        assert _normalize_name("  TP53  ") == "tp53"

    def test_lowercases(self) -> None:
        assert _normalize_name("EGFR") == "egfr"

    def test_empty_string(self) -> None:
        assert _normalize_name("") == ""


# ------------------------------------------------------------------
# BioEntityExtractor.extract_entities tests
# ------------------------------------------------------------------


class TestExtractEntities:
    """Test entity extraction with mocked LLM."""

    async def test_extracts_entities_from_llm_response(
        self, extractor: BioEntityExtractor
    ) -> None:
        llm_response = json.dumps(
            [
                {"name": "KRAS", "entity_type": "Gene", "aliases": ["K-Ras"], "identifiers": {}},
                {
                    "name": "lung cancer",
                    "entity_type": "Disease",
                    "aliases": ["NSCLC"],
                    "identifiers": {},
                },
            ]
        )
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        entities = await extractor.extract_entities("KRAS mutations in lung cancer")

        assert len(entities) == 2
        assert entities[0].name == "kras"
        assert entities[0].entity_type == "Gene"
        assert entities[1].name == "lung cancer"

    async def test_handles_markdown_wrapped_json(self, extractor: BioEntityExtractor) -> None:
        llm_response = '```json\n[{"name": "TP53", "entity_type": "Gene"}]\n```'
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        entities = await extractor.extract_entities("TP53 is a tumor suppressor")
        assert len(entities) == 1
        assert entities[0].name == "tp53"

    async def test_filters_invalid_entity_types(self, extractor: BioEntityExtractor) -> None:
        llm_response = json.dumps(
            [
                {"name": "TP53", "entity_type": "Gene"},
                {"name": "something", "entity_type": "InvalidType"},
            ]
        )
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        entities = await extractor.extract_entities("test text")
        assert len(entities) == 1
        assert entities[0].entity_type == "Gene"

    async def test_fallback_on_first_failure(self, extractor: BioEntityExtractor) -> None:
        good_response = json.dumps([{"name": "EGFR", "entity_type": "Protein"}])
        extractor._ollama.generate = AsyncMock(side_effect=[Exception("LLM error"), good_response])

        entities = await extractor.extract_entities("EGFR in cancer")
        assert len(entities) == 1
        assert entities[0].name == "egfr"

    async def test_returns_empty_on_total_failure(self, extractor: BioEntityExtractor) -> None:
        extractor._ollama.generate = AsyncMock(side_effect=Exception("LLM down"))

        entities = await extractor.extract_entities("some text")
        assert entities == []

    async def test_handles_dict_wrapped_response(self, extractor: BioEntityExtractor) -> None:
        llm_response = json.dumps({"entities": [{"name": "aspirin", "entity_type": "Drug"}]})
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        entities = await extractor.extract_entities("aspirin for pain")
        assert len(entities) == 1
        assert entities[0].name == "aspirin"


# ------------------------------------------------------------------
# BioEntityExtractor.extract_relations tests
# ------------------------------------------------------------------


class TestExtractRelations:
    """Test relation extraction with mocked LLM."""

    async def test_extracts_relations(self, extractor: BioEntityExtractor) -> None:
        entities = [
            BioEntity(name="kras", entity_type="Gene"),
            BioEntity(name="lung cancer", entity_type="Disease"),
        ]
        llm_response = json.dumps(
            [
                {
                    "source": "kras",
                    "target": "lung cancer",
                    "relation_type": "ASSOCIATED_WITH",
                    "confidence": 0.9,
                    "evidence": "KRAS mutations drive lung cancer",
                }
            ]
        )
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        relations = await extractor.extract_relations("KRAS in lung cancer", entities)

        assert len(relations) == 1
        assert relations[0].source == "kras"
        assert relations[0].relation_type == "ASSOCIATED_WITH"
        assert relations[0].confidence == 0.9

    async def test_filters_invalid_relation_types(self, extractor: BioEntityExtractor) -> None:
        entities = [
            BioEntity(name="a", entity_type="Gene"),
            BioEntity(name="b", entity_type="Disease"),
        ]
        llm_response = json.dumps(
            [
                {
                    "source": "a",
                    "target": "b",
                    "relation_type": "ASSOCIATED_WITH",
                    "confidence": 0.8,
                    "evidence": "x",
                },
                {
                    "source": "a",
                    "target": "b",
                    "relation_type": "INVALID_REL",
                    "confidence": 0.5,
                    "evidence": "y",
                },
            ]
        )
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        relations = await extractor.extract_relations("text", entities)
        assert len(relations) == 1

    async def test_skips_self_relations(self, extractor: BioEntityExtractor) -> None:
        entities = [
            BioEntity(name="tp53", entity_type="Gene"),
            BioEntity(name="brca1", entity_type="Gene"),
        ]
        llm_response = json.dumps(
            [
                {
                    "source": "tp53",
                    "target": "tp53",
                    "relation_type": "INTERACTS_WITH",
                    "confidence": 0.5,
                    "evidence": "x",
                },
            ]
        )
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        relations = await extractor.extract_relations("text", entities)
        assert len(relations) == 0

    async def test_skips_entities_not_in_list(self, extractor: BioEntityExtractor) -> None:
        entities = [BioEntity(name="tp53", entity_type="Gene")]
        llm_response = json.dumps(
            [
                {
                    "source": "tp53",
                    "target": "unknown_entity",
                    "relation_type": "TARGETS",
                    "confidence": 0.5,
                    "evidence": "x",
                }
            ]
        )
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        relations = await extractor.extract_relations("text", entities)
        assert len(relations) == 0

    async def test_returns_empty_for_single_entity(self, extractor: BioEntityExtractor) -> None:
        entities = [BioEntity(name="tp53", entity_type="Gene")]
        relations = await extractor.extract_relations("text", entities)
        assert relations == []

    async def test_clamps_confidence(self, extractor: BioEntityExtractor) -> None:
        entities = [
            BioEntity(name="a", entity_type="Gene"),
            BioEntity(name="b", entity_type="Disease"),
        ]
        llm_response = json.dumps(
            [
                {
                    "source": "a",
                    "target": "b",
                    "relation_type": "CAUSES",
                    "confidence": 5.0,
                    "evidence": "x",
                },
            ]
        )
        extractor._ollama.generate = AsyncMock(return_value=llm_response)

        relations = await extractor.extract_relations("text", entities)
        assert relations[0].confidence == 1.0
