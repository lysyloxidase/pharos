"""Tests for pharos.orchestration.task_models — Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pharos.orchestration.task_models import AgentResult, Task, TaskType, WorkflowState


class TestTask:
    """Test Task model validation."""

    def test_minimal_task(self) -> None:
        task = Task(query="What is TP53?")
        assert task.query == "What is TP53?"
        assert task.task_type is None
        assert task.context == {}
        assert task.parent_task_id is None

    def test_task_with_type(self) -> None:
        task = Task(query="Predict approval", task_type=TaskType.FORECAST)
        assert task.task_type == TaskType.FORECAST

    def test_task_with_context(self) -> None:
        task = Task(query="test", context={"key": "value"})
        assert task.context["key"] == "value"

    def test_task_requires_query(self) -> None:
        with pytest.raises(ValidationError):
            Task()  # type: ignore[call-arg]

    def test_task_type_from_string(self) -> None:
        task = Task(query="test", task_type="forecast")  # type: ignore[arg-type]
        assert task.task_type == TaskType.FORECAST

    def test_invalid_task_type(self) -> None:
        with pytest.raises(ValidationError):
            Task(query="test", task_type="invalid_type")  # type: ignore[arg-type]


class TestAgentResult:
    """Test AgentResult model validation."""

    def test_minimal_result(self) -> None:
        result = AgentResult(
            agent_name="Oracle",
            task_id="abc-123",
            content="Some output",
        )
        assert result.agent_name == "Oracle"
        assert result.confidence == 0.0
        assert result.artifacts == {}
        assert result.kg_updates == []

    def test_result_with_all_fields(self) -> None:
        result = AgentResult(
            agent_name="Alchemist",
            task_id="xyz-789",
            content="Designed molecule",
            artifacts={"smiles": "CCO"},
            confidence=0.92,
            kg_updates=[{"source": "Drug", "relation": "TARGETS", "target": "Gene"}],
        )
        assert result.confidence == 0.92
        assert result.artifacts["smiles"] == "CCO"
        assert len(result.kg_updates) == 1

    def test_result_requires_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentResult()  # type: ignore[call-arg]


class TestTaskType:
    """Test TaskType enum."""

    def test_all_values(self) -> None:
        expected = {
            "forecast",
            "review",
            "build_kg",
            "design_molecule",
            "design_protein",
            "verify",
            "general",
        }
        assert {t.value for t in TaskType} == expected

    def test_string_comparison(self) -> None:
        assert TaskType.FORECAST == "forecast"
        assert TaskType.GENERAL == "general"


class TestWorkflowState:
    """Test WorkflowState TypedDict construction."""

    def test_create_state(self) -> None:
        task = Task(query="test")
        state: WorkflowState = {
            "task": task,
            "results": [],
            "current_agent": "router",
            "kg_context": "",
            "iteration": 0,
        }
        assert state["task"].query == "test"
        assert state["iteration"] == 0
