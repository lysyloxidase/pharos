"""Tests for pharos.agents.oracle — forecasting pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.agents.oracle import OracleAgent
from pharos.config import Settings
from pharos.forecasting.hypothesis_gen import Hypothesis, HypothesisGenerator
from pharos.forecasting.trend_detector import EntityTrend, TrendDetector
from pharos.orchestration.task_models import Task, TaskType, WorkflowState

# ------------------------------------------------------------------
# Sample data
# ------------------------------------------------------------------

SAMPLE_ENTITIES_JSON = json.dumps(["KRAS", "lung cancer"])

SAMPLE_HYPOTHESES_JSON = json.dumps(
    [
        {
            "statement": "KRAS G12D may be targetable via novel allosteric inhibitors for pancreatic cancer",
            "target_entity": "KRAS",
            "disease_entity": "pancreatic cancer",
            "evidence": ["Rising publication trend", "KG path exists"],
            "confidence": 0.75,
            "time_horizon": "3-5 years",
            "kg_path": ["KRAS", "MAPK pathway", "pancreatic cancer"],
        },
        {
            "statement": "CDK12 inhibition may synergize with immunotherapy in NSCLC",
            "target_entity": "CDK12",
            "disease_entity": "NSCLC",
            "evidence": ["Converging publication patterns"],
            "confidence": 0.60,
            "time_horizon": "1-2 years",
        },
    ]
)

SAMPLE_REPORT = (
    "# Forecasting Report\n\n"
    "## Executive Summary\n"
    "KRAS-targeted therapies show accelerating interest.\n\n"
    "## Trends\n"
    "KRAS publications growing at 10 pub/yr.\n\n"
    "## Hypotheses\n"
    "1. KRAS G12D may be targetable.\n"
    "2. CDK12 inhibition may synergize.\n"
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.oracle_max_pubmed_queries = 20
    s.oracle_trend_years_start = 2022
    s.oracle_trend_years_end = 2025
    return s


@pytest.fixture
def mock_trend_detector() -> MagicMock:
    det = MagicMock(spec=TrendDetector)
    det.compute_entity_trends = AsyncMock(
        return_value=EntityTrend(
            entity="KRAS",
            yearly_counts={2022: 100, 2023: 130, 2024: 170},
            velocity=35.0,
            acceleration=5.0,
            semantic_drift=0.15,
            emerging_associations=[("sotorasib", 0.3), ("adagrasib", 0.2)],
        )
    )
    det.find_converging_entities = AsyncMock()
    det.reset_query_count = MagicMock()
    return det


@pytest.fixture
def mock_hypothesis_gen() -> MagicMock:
    gen = MagicMock(spec=HypothesisGenerator)
    gen.generate_hypotheses = AsyncMock(
        return_value=[
            Hypothesis(
                statement="KRAS G12D may be targetable",
                target_entity="KRAS",
                disease_entity="pancreatic cancer",
                evidence=["Rising trend"],
                confidence=0.75,
                time_horizon="3-5 years",
            ),
            Hypothesis(
                statement="CDK12 may synergize with IO",
                target_entity="CDK12",
                disease_entity="NSCLC",
                evidence=["Convergence signal"],
                confidence=0.60,
                time_horizon="1-2 years",
            ),
        ]
    )
    return gen


@pytest.fixture
def oracle(
    mock_ollama: MagicMock,
    mock_neo4j: MagicMock,
    settings: Settings,
    mock_trend_detector: MagicMock,
    mock_hypothesis_gen: MagicMock,
) -> OracleAgent:
    return OracleAgent(
        ollama=mock_ollama,
        kg=mock_neo4j,
        config=settings,
        trend_detector=mock_trend_detector,
        hypothesis_gen=mock_hypothesis_gen,
    )


@pytest.fixture
def state() -> WorkflowState:
    return WorkflowState(
        task=Task(query="test", task_type=TaskType.FORECAST),
        results=[],
        current_agent="oracle",
        kg_context="",
        iteration=0,
    )


# ------------------------------------------------------------------
# Entity extraction tests
# ------------------------------------------------------------------


class TestOracleEntityExtraction:
    """Test entity extraction from forecasting queries."""

    async def test_extracts_entities(self, oracle: OracleAgent) -> None:
        oracle.ollama.generate = AsyncMock(return_value=SAMPLE_ENTITIES_JSON)
        entities = await oracle._extract_entities("Predict KRAS targets in lung cancer")
        assert "KRAS" in entities
        assert "lung cancer" in entities

    async def test_handles_dict_wrapped(self, oracle: OracleAgent) -> None:
        oracle.ollama.generate = AsyncMock(
            return_value=json.dumps({"entities": ["TP53", "breast cancer"]})
        )
        entities = await oracle._extract_entities("TP53 in breast cancer")
        assert "TP53" in entities

    async def test_fallback_on_failure(self, oracle: OracleAgent) -> None:
        oracle.ollama.generate = AsyncMock(side_effect=Exception("LLM down"))
        entities = await oracle._extract_entities("some query")
        assert entities == []


# ------------------------------------------------------------------
# Full pipeline tests
# ------------------------------------------------------------------


class TestOracleRun:
    """Test the full Oracle pipeline."""

    async def test_full_pipeline(
        self,
        oracle: OracleAgent,
        state: WorkflowState,
        mock_trend_detector: MagicMock,
        mock_hypothesis_gen: MagicMock,
    ) -> None:
        oracle.ollama.generate = AsyncMock(
            side_effect=[
                SAMPLE_ENTITIES_JSON,  # entity extraction
                SAMPLE_REPORT,  # report generation
            ]
        )

        task = Task(query="Predict emerging KRAS therapeutic targets")
        result = await oracle.run(task, state)

        assert result.agent_name == "Oracle"
        assert result.content == SAMPLE_REPORT
        assert len(result.artifacts["hypotheses"]) == 2
        assert len(result.artifacts["trends"]) > 0
        assert result.confidence > 0

        mock_trend_detector.compute_entity_trends.assert_called()
        mock_hypothesis_gen.generate_hypotheses.assert_called_once()

    async def test_uses_query_as_fallback_entity(
        self, oracle: OracleAgent, state: WorkflowState
    ) -> None:
        oracle.ollama.generate = AsyncMock(
            side_effect=[
                "invalid json",  # entity extraction fails
                SAMPLE_REPORT,
            ]
        )

        task = Task(query="novel targets")
        result = await oracle.run(task, state)

        assert result.agent_name == "Oracle"
        # Should still work with the whole query as entity
        assert "novel targets" in result.artifacts["entities_analysed"]

    async def test_confidence_from_hypotheses(
        self, oracle: OracleAgent, state: WorkflowState
    ) -> None:
        oracle.ollama.generate = AsyncMock(side_effect=[SAMPLE_ENTITIES_JSON, SAMPLE_REPORT])

        task = Task(query="KRAS forecast")
        result = await oracle.run(task, state)

        # Average of 0.75 and 0.60 = 0.675
        assert result.confidence == pytest.approx(0.675)

    async def test_low_confidence_without_hypotheses(
        self,
        oracle: OracleAgent,
        mock_hypothesis_gen: MagicMock,
        state: WorkflowState,
    ) -> None:
        mock_hypothesis_gen.generate_hypotheses = AsyncMock(return_value=[])
        oracle.ollama.generate = AsyncMock(side_effect=[SAMPLE_ENTITIES_JSON, "Empty report."])

        task = Task(query="obscure topic")
        result = await oracle.run(task, state)

        assert result.confidence == pytest.approx(0.2)

    async def test_limits_entities_to_five(
        self, oracle: OracleAgent, mock_trend_detector: MagicMock, state: WorkflowState
    ) -> None:
        many_entities = json.dumps(["A", "B", "C", "D", "E", "F", "G"])
        oracle.ollama.generate = AsyncMock(side_effect=[many_entities, SAMPLE_REPORT])

        task = Task(query="broad query")
        result = await oracle.run(task, state)

        assert len(result.artifacts["entities_analysed"]) == 5
        assert mock_trend_detector.compute_entity_trends.call_count == 5


# ------------------------------------------------------------------
# Report generation tests
# ------------------------------------------------------------------


class TestOracleReport:
    """Test report generation."""

    async def test_generates_report(self, oracle: OracleAgent) -> None:
        oracle.ollama.generate = AsyncMock(return_value="Report text here.")
        trend = EntityTrend(entity="KRAS", yearly_counts={2023: 100}, velocity=10.0)
        hyp = Hypothesis(
            statement="Test hypothesis",
            confidence=0.8,
            time_horizon="1-2 years",
        )
        report = await oracle._generate_report("test query", [trend], [hyp])
        assert report == "Report text here."

    async def test_report_prompt_includes_trends(self, oracle: OracleAgent) -> None:
        oracle.ollama.generate = AsyncMock(return_value="report")
        trend = EntityTrend(
            entity="TP53",
            yearly_counts={2023: 50, 2024: 70},
            velocity=20.0,
            acceleration=3.0,
            semantic_drift=0.1,
            emerging_associations=[("MDM2", 0.4)],
        )
        await oracle._generate_report("TP53 forecast", [trend], [])

        call_args = oracle.ollama.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt", "")
        assert "TP53" in prompt
        assert "20.0" in prompt
