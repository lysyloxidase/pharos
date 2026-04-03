"""Tests for pharos.agents.router — task classification with mock Ollama."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.agents.router import _KEYWORD_MAP, RouterAgent
from pharos.config import Settings
from pharos.orchestration.task_models import Task, TaskType, WorkflowState


@pytest.fixture
def router(mock_ollama: MagicMock, mock_neo4j: MagicMock) -> RouterAgent:
    """Create a RouterAgent with mocked dependencies."""
    return RouterAgent(
        ollama=mock_ollama,
        kg=mock_neo4j,
        config=Settings(),
    )


@pytest.fixture
def state() -> WorkflowState:
    """Create an initial workflow state."""
    return WorkflowState(
        task=Task(query="test"),
        results=[],
        current_agent="router",
        kg_context="",
        iteration=0,
    )


class TestRouterClassification:
    """Test LLM-based task classification."""

    async def test_classifies_forecast(self, router: RouterAgent, state: WorkflowState) -> None:
        router.ollama.generate = AsyncMock(return_value=json.dumps({"task_type": "forecast"}))
        task = Task(query="Will CRISPR therapies be approved by 2026?")
        result = await router.run(task, state)
        assert result.artifacts["task_type"] == "forecast"
        assert task.task_type == TaskType.FORECAST

    async def test_classifies_review(self, router: RouterAgent, state: WorkflowState) -> None:
        router.ollama.generate = AsyncMock(return_value=json.dumps({"task_type": "review"}))
        task = Task(query="Write a review of CAR-T therapy")
        result = await router.run(task, state)
        assert result.artifacts["task_type"] == "review"

    async def test_classifies_design_molecule(
        self, router: RouterAgent, state: WorkflowState
    ) -> None:
        router.ollama.generate = AsyncMock(
            return_value=json.dumps({"task_type": "design_molecule"})
        )
        task = Task(query="Design a JAK2 inhibitor")
        result = await router.run(task, state)
        assert result.artifacts["task_type"] == "design_molecule"

    async def test_handles_sub_tasks(self, router: RouterAgent, state: WorkflowState) -> None:
        router.ollama.generate = AsyncMock(
            return_value=json.dumps(
                {
                    "task_type": "review",
                    "sub_tasks": ["Search literature", "Summarize findings"],
                }
            )
        )
        task = Task(query="Complex multi-step query")
        result = await router.run(task, state)
        assert result.artifacts["sub_tasks"] == [
            "Search literature",
            "Summarize findings",
        ]


class TestRouterFallback:
    """Test keyword-based fallback when LLM fails."""

    async def test_fallback_on_llm_error(self, router: RouterAgent, state: WorkflowState) -> None:
        router.ollama.generate = AsyncMock(side_effect=Exception("LLM down"))
        task = Task(query="Predict the next breakthrough in Alzheimer's")
        await router.run(task, state)
        # "predict" keyword maps to FORECAST
        assert task.task_type == TaskType.FORECAST

    async def test_fallback_to_general(self, router: RouterAgent, state: WorkflowState) -> None:
        router.ollama.generate = AsyncMock(side_effect=Exception("LLM down"))
        task = Task(query="Tell me something interesting")
        await router.run(task, state)
        assert task.task_type == TaskType.GENERAL


class TestKeywordMap:
    """Test the keyword fallback mapping."""

    def test_keyword_map_covers_all_types(self) -> None:
        covered_types = set(_KEYWORD_MAP.values())
        # GENERAL is the default fallback, doesn't need a keyword
        expected = set(TaskType) - {TaskType.GENERAL}
        assert expected.issubset(covered_types)

    def test_keyword_matching(self) -> None:
        from pharos.agents.router import RouterAgent

        assert RouterAgent._keyword_fallback("predict drug approval") == TaskType.FORECAST
        assert RouterAgent._keyword_fallback("write a review") == TaskType.REVIEW
        assert RouterAgent._keyword_fallback("build knowledge graph") == TaskType.BUILD_KG
        assert RouterAgent._keyword_fallback("verify this claim") == TaskType.VERIFY
