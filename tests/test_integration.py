"""Integration tests — full pipeline flows with mocked externals."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from pharos.orchestration.graph_workflow import (
    _format_final_output,
    _route_task,
    _sentinel_post_check,
    build_workflow,
)
from pharos.orchestration.task_models import AgentResult, Task, TaskType, WorkflowState

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_agent(name: str, content: str = "", confidence: float = 0.8) -> MagicMock:
    """Create a mock agent returning a predictable AgentResult."""
    agent = MagicMock()
    agent.run = AsyncMock(
        return_value=AgentResult(
            agent_name=name,
            task_id=str(uuid.uuid4()),
            content=content or f"{name} result content.",
            confidence=confidence,
        )
    )
    return agent


def _make_state(
    task_type: TaskType = TaskType.FORECAST,
    query: str = "Predict KRAS targets",
) -> WorkflowState:
    return WorkflowState(
        task=Task(query=query, task_type=task_type),
        results=[],
        current_agent="router",
        kg_context="",
        iteration=0,
    )


def _mock_agents(overrides: dict | None = None) -> dict[str, MagicMock]:
    agents = {
        "router": _make_mock_agent("router"),
        "oracle": _make_mock_agent("Oracle", "KRAS is trending.", 0.75),
        "scribe": _make_mock_agent("Scribe", "Review of KRAS therapies.", 0.8),
        "cartographer": _make_mock_agent("Cartographer", "Built KG.", 0.7),
        "alchemist": _make_mock_agent("Alchemist", "Designed molecule.", 0.6),
        "architect": _make_mock_agent("Architect", "Designed protein.", 0.7),
        "sentinel": _make_mock_agent("Sentinel", "Verification report.", 0.9),
    }
    if overrides:
        agents.update(overrides)
    return agents


# ------------------------------------------------------------------
# Router → Agent flow tests
# ------------------------------------------------------------------


class TestRouterToAgentFlow:
    """Verify routing works for each TaskType."""

    def test_forecast_routes_to_oracle(self) -> None:
        state = _make_state(TaskType.FORECAST)
        assert _route_task(state) == "oracle"

    def test_review_routes_to_scribe(self) -> None:
        state = _make_state(TaskType.REVIEW)
        assert _route_task(state) == "scribe"

    def test_build_kg_routes_to_cartographer(self) -> None:
        state = _make_state(TaskType.BUILD_KG)
        assert _route_task(state) == "cartographer"

    def test_design_molecule_routes_to_alchemist(self) -> None:
        state = _make_state(TaskType.DESIGN_MOLECULE)
        assert _route_task(state) == "alchemist"

    def test_design_protein_routes_to_architect(self) -> None:
        state = _make_state(TaskType.DESIGN_PROTEIN)
        assert _route_task(state) == "architect"

    def test_verify_routes_to_sentinel(self) -> None:
        state = _make_state(TaskType.VERIFY)
        assert _route_task(state) == "sentinel"

    def test_general_routes_to_general(self) -> None:
        state = _make_state(TaskType.GENERAL)
        assert _route_task(state) == "general"

    def test_none_routes_to_general(self) -> None:
        state = WorkflowState(
            task=Task(query="test", task_type=None),
            results=[],
            current_agent="router",
            kg_context="",
            iteration=0,
        )
        assert _route_task(state) == "general"


# ------------------------------------------------------------------
# Sentinel retry logic
# ------------------------------------------------------------------


class TestSentinelRetryLogic:
    def test_proceeds_to_aggregate_on_high_score(self) -> None:
        state = _make_state()
        state["results"] = [
            AgentResult(agent_name="Sentinel", task_id="t1", content="ok", confidence=0.8)
        ]
        assert _sentinel_post_check(state) == "aggregate"

    def test_retries_on_low_score(self) -> None:
        state = _make_state(TaskType.FORECAST)
        state["results"] = [
            AgentResult(agent_name="Sentinel", task_id="t1", content="bad", confidence=0.3)
        ]
        assert _sentinel_post_check(state) == "oracle"

    def test_stops_after_max_retries(self) -> None:
        state = _make_state(TaskType.FORECAST)
        state["results"] = [
            AgentResult(agent_name="Sentinel", task_id="t1", content="bad", confidence=0.3),
            AgentResult(agent_name="Sentinel", task_id="t2", content="still bad", confidence=0.4),
        ]
        assert _sentinel_post_check(state) == "aggregate"

    def test_no_sentinel_results_goes_to_aggregate(self) -> None:
        state = _make_state()
        state["results"] = []
        assert _sentinel_post_check(state) == "aggregate"


# ------------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------------


class TestFormatFinalOutput:
    def test_formats_specialist_results(self) -> None:
        results = [
            AgentResult(agent_name="router", task_id="t0", content="routed", confidence=1.0),
            AgentResult(
                agent_name="Oracle", task_id="t1", content="KRAS analysis.", confidence=0.75
            ),
            AgentResult(
                agent_name="Sentinel",
                task_id="t2",
                content="# Verification\nAll good.",
                confidence=0.9,
            ),
        ]
        output = _format_final_output(results)

        assert "# PHAROS Results" in output
        assert "Oracle" in output
        assert "KRAS analysis." in output
        assert "Verification" in output
        # Router should NOT appear as a section
        assert "## router" not in output

    def test_deduplicates_agents(self) -> None:
        results = [
            AgentResult(agent_name="Oracle", task_id="t1", content="First run.", confidence=0.5),
            AgentResult(agent_name="Oracle", task_id="t2", content="Second run.", confidence=0.8),
        ]
        output = _format_final_output(results)
        assert output.count("## Oracle") == 1

    def test_empty_results(self) -> None:
        output = _format_final_output([])
        assert "PHAROS Results" in output


# ------------------------------------------------------------------
# Full pipeline (mock workflow execution)
# ------------------------------------------------------------------


class TestFullPipelineForecast:
    """Test full forecast pipeline: Router → Oracle → Sentinel → Aggregate."""

    async def test_forecast_pipeline(self) -> None:
        agents = _mock_agents()
        build_workflow(agents)  # ensure graph builds without error

        state = _make_state(TaskType.FORECAST, "Predict KRAS therapeutic targets")

        # We can't easily run LangGraph's compiled graph in tests without
        # full LangGraph runtime, so test the individual steps.
        # 1. Router
        router_result = await agents["router"].run(state["task"], state)
        state["results"].append(router_result)

        # 2. Oracle (routed by _route_task)
        agent_name = _route_task(state)
        assert agent_name == "oracle"
        oracle_result = await agents[agent_name].run(state["task"], state)
        state["results"].append(oracle_result)

        # 3. Sentinel
        sentinel_result = await agents["sentinel"].run(state["task"], state)
        state["results"].append(sentinel_result)

        # 4. Check post-sentinel routing
        next_node = _sentinel_post_check(state)
        assert next_node == "aggregate"

        # 5. Verify output format
        output = _format_final_output(state["results"])
        assert "Oracle" in output
        assert "KRAS is trending" in output


class TestFullPipelineReview:
    """Test full review pipeline: Router → Scribe → Sentinel → Aggregate."""

    async def test_review_pipeline(self) -> None:
        agents = _mock_agents()
        state = _make_state(TaskType.REVIEW, "Review CAR-T therapy advances")

        agent_name = _route_task(state)
        assert agent_name == "scribe"

        scribe_result = await agents["scribe"].run(state["task"], state)
        state["results"].append(scribe_result)

        sentinel_result = await agents["sentinel"].run(state["task"], state)
        state["results"].append(sentinel_result)

        output = _format_final_output(state["results"])
        assert "Scribe" in output
        assert "Review of KRAS therapies" in output


class TestFullPipelineProtein:
    """Test protein design pipeline: Router → Architect → Sentinel."""

    async def test_protein_pipeline(self) -> None:
        agents = _mock_agents()
        state = _make_state(TaskType.DESIGN_PROTEIN, "Design thermostable lipase")

        agent_name = _route_task(state)
        assert agent_name == "architect"

        arch_result = await agents["architect"].run(state["task"], state)
        state["results"].append(arch_result)

        sentinel_result = await agents["sentinel"].run(state["task"], state)
        state["results"].append(sentinel_result)

        next_node = _sentinel_post_check(state)
        assert next_node == "aggregate"


class TestFullPipelineRetry:
    """Test retry flow when Sentinel scores low."""

    async def test_retry_on_low_confidence(self) -> None:
        low_sentinel = _make_mock_agent("Sentinel", "Issues found.", 0.3)
        agents = _mock_agents({"sentinel": low_sentinel})

        state = _make_state(TaskType.FORECAST, "Predict targets")

        # Oracle run
        oracle_result = await agents["oracle"].run(state["task"], state)
        state["results"].append(oracle_result)

        # Sentinel run (low confidence)
        sentinel_result = await agents["sentinel"].run(state["task"], state)
        state["results"].append(sentinel_result)

        # Should retry
        next_node = _sentinel_post_check(state)
        assert next_node == "oracle"

        # Second Oracle run
        oracle_result2 = await agents["oracle"].run(state["task"], state)
        state["results"].append(oracle_result2)

        # Second Sentinel run
        sentinel_result2 = await agents["sentinel"].run(state["task"], state)
        state["results"].append(sentinel_result2)

        # Should stop now (max retries)
        next_node = _sentinel_post_check(state)
        assert next_node == "aggregate"


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------


class TestBuildWorkflow:
    def test_builds_successfully(self) -> None:
        agents = _mock_agents()
        graph = build_workflow(agents)
        assert graph is not None

    def test_has_all_nodes(self) -> None:
        agents = _mock_agents()
        graph = build_workflow(agents)
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
