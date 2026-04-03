"""Tests for pharos.orchestration.graph_workflow — workflow routing and structure."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.orchestration.graph_workflow import (
    _format_final_output,
    _route_task,
    _sentinel_post_check,
    build_workflow,
)
from pharos.orchestration.task_models import AgentResult, Task, TaskType, WorkflowState


def _make_mock_agent(name: str) -> MagicMock:
    """Create a mock agent that returns a predictable AgentResult."""
    agent = MagicMock()
    agent.run = AsyncMock(
        return_value=AgentResult(
            agent_name=name,
            task_id=str(uuid.uuid4()),
            content=f"{name} result",
            confidence=0.8,
        )
    )
    return agent


@pytest.fixture
def mock_agents() -> dict[str, MagicMock]:
    """Create a full set of mock agents."""
    return {
        "router": _make_mock_agent("router"),
        "oracle": _make_mock_agent("oracle"),
        "scribe": _make_mock_agent("scribe"),
        "cartographer": _make_mock_agent("cartographer"),
        "alchemist": _make_mock_agent("alchemist"),
        "architect": _make_mock_agent("architect"),
        "sentinel": _make_mock_agent("sentinel"),
    }


class TestRouteTask:
    """Test the conditional routing function."""

    def test_routes_forecast_to_oracle(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.FORECAST),
            results=[],
            current_agent="router",
            kg_context="",
            iteration=0,
        )
        assert _route_task(state) == "oracle"

    def test_routes_review_to_scribe(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.REVIEW),
            results=[],
            current_agent="router",
            kg_context="",
            iteration=0,
        )
        assert _route_task(state) == "scribe"

    def test_routes_build_kg_to_cartographer(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.BUILD_KG),
            results=[],
            current_agent="router",
            kg_context="",
            iteration=0,
        )
        assert _route_task(state) == "cartographer"

    def test_routes_design_molecule_to_alchemist(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.DESIGN_MOLECULE),
            results=[],
            current_agent="router",
            kg_context="",
            iteration=0,
        )
        assert _route_task(state) == "alchemist"

    def test_routes_design_protein_to_architect(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.DESIGN_PROTEIN),
            results=[],
            current_agent="router",
            kg_context="",
            iteration=0,
        )
        assert _route_task(state) == "architect"

    def test_routes_verify_to_sentinel(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.VERIFY),
            results=[],
            current_agent="router",
            kg_context="",
            iteration=0,
        )
        assert _route_task(state) == "sentinel"

    def test_routes_none_to_general(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=None),
            results=[],
            current_agent="router",
            kg_context="",
            iteration=0,
        )
        assert _route_task(state) == "general"


class TestSentinelPostCheck:
    """Test the sentinel retry/aggregate decision."""

    def test_aggregates_on_high_score(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.FORECAST),
            results=[
                AgentResult(agent_name="Sentinel", task_id="t1", content="ok", confidence=0.8)
            ],
            current_agent="sentinel",
            kg_context="",
            iteration=0,
        )
        assert _sentinel_post_check(state) == "aggregate"

    def test_retries_on_low_score(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.FORECAST),
            results=[
                AgentResult(agent_name="Sentinel", task_id="t1", content="bad", confidence=0.3)
            ],
            current_agent="sentinel",
            kg_context="",
            iteration=0,
        )
        assert _sentinel_post_check(state) == "oracle"

    def test_stops_after_max_retries(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=TaskType.FORECAST),
            results=[
                AgentResult(agent_name="Sentinel", task_id="t1", content="bad", confidence=0.3),
                AgentResult(agent_name="Sentinel", task_id="t2", content="bad", confidence=0.4),
            ],
            current_agent="sentinel",
            kg_context="",
            iteration=0,
        )
        assert _sentinel_post_check(state) == "aggregate"


class TestFormatFinalOutput:
    """Test Markdown output formatting."""

    def test_includes_specialist_content(self) -> None:
        results = [
            AgentResult(agent_name="Oracle", task_id="t1", content="KRAS data.", confidence=0.8),
        ]
        output = _format_final_output(results)
        assert "Oracle" in output
        assert "KRAS data." in output

    def test_excludes_router(self) -> None:
        results = [
            AgentResult(agent_name="router", task_id="t0", content="routed", confidence=1.0),
            AgentResult(agent_name="Oracle", task_id="t1", content="data", confidence=0.8),
        ]
        output = _format_final_output(results)
        assert "## router" not in output


class TestBuildWorkflow:
    """Test workflow graph construction."""

    def test_build_returns_state_graph(self, mock_agents: dict[str, MagicMock]) -> None:
        graph = build_workflow(mock_agents)
        assert graph is not None

    def test_graph_has_all_nodes(self, mock_agents: dict[str, MagicMock]) -> None:
        graph = build_workflow(mock_agents)
        node_names = set(graph.nodes.keys())
        expected = {
            "route",
            "oracle",
            "scribe",
            "cartographer",
            "alchemist",
            "architect",
            "sentinel",
            "aggregate",
        }
        assert expected.issubset(node_names)
