"""LangGraph StateGraph definition — the main PHAROS workflow.

Defines the multi-agent orchestration graph with conditional routing,
mandatory verification (Sentinel), retry on low confidence, and result
aggregation with Markdown formatting.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from pharos.orchestration.task_models import AgentResult, TaskType, WorkflowState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node functions — each wraps an agent's ``run()`` method
# ---------------------------------------------------------------------------


def _make_agent_node(agent_name: str) -> Any:
    """Create a LangGraph node function for a given agent.

    The actual agent instances are injected at graph-compile time via
    ``build_workflow``. This factory returns a closure that looks up the
    agent from a registry and invokes it.

    Args:
        agent_name: Logical agent name (e.g. "oracle").

    Returns:
        An async node function compatible with LangGraph.
    """

    async def _node(state: WorkflowState) -> WorkflowState:
        agent = _AGENT_REGISTRY.get(agent_name)
        if agent is None:
            logger.warning("Agent '%s' not registered, returning stub result", agent_name)
            state["results"].append(
                AgentResult(
                    agent_name=agent_name,
                    task_id=str(uuid.uuid4()),
                    content=f"[{agent_name}] not registered",
                )
            )
            return state

        state["current_agent"] = agent_name
        result = await agent.run(state["task"], state)
        state["results"].append(result)
        return state

    _node.__name__ = agent_name
    return _node


# Global registry populated by ``build_workflow``
_AGENT_REGISTRY: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Router edge — conditional routing based on task_type
# ---------------------------------------------------------------------------


def _route_task(state: WorkflowState) -> str:
    """Determine the next node based on the classified task type.

    Args:
        state: Current workflow state (task.task_type set by router).

    Returns:
        Name of the next agent node.
    """
    task_type = state["task"].task_type

    routing_map: dict[TaskType, str] = {
        TaskType.FORECAST: "oracle",
        TaskType.REVIEW: "scribe",
        TaskType.BUILD_KG: "cartographer",
        TaskType.DESIGN_MOLECULE: "alchemist",
        TaskType.DESIGN_PROTEIN: "architect",
        TaskType.VERIFY: "sentinel",
        TaskType.GENERAL: "general",
    }

    if task_type is None:
        return "general"

    return routing_map.get(task_type, "general")


# ---------------------------------------------------------------------------
# Sentinel → decide retry or aggregate
# ---------------------------------------------------------------------------

_MAX_RETRIES = 1


def _sentinel_post_check(state: WorkflowState) -> str:
    """After Sentinel verifies, decide whether to retry or aggregate.

    If the Sentinel's verification score is below 0.5 and we haven't
    retried yet, route back to the specialist agent for a second attempt.
    Otherwise proceed to aggregation.

    Args:
        state: Current workflow state.

    Returns:
        Name of the next node ("aggregate" or the specialist agent).
    """
    sentinel_results = [r for r in state["results"] if r.agent_name.lower() == "sentinel"]
    retry_count = len(sentinel_results)

    if retry_count > _MAX_RETRIES:
        return "aggregate"

    # Check last sentinel score
    if sentinel_results:
        last_sentinel = sentinel_results[-1]
        if last_sentinel.confidence < 0.5:
            # Find which specialist ran
            specialist = _route_task(state)
            if specialist != "sentinel":
                logger.info(
                    "Sentinel score %.2f < 0.5 — retrying %s (attempt %d)",
                    last_sentinel.confidence,
                    specialist,
                    retry_count + 1,
                )
                return specialist

    return "aggregate"


# ---------------------------------------------------------------------------
# Aggregate node
# ---------------------------------------------------------------------------


async def _aggregate(state: WorkflowState) -> WorkflowState:
    """Collect all agent results, format output, and prepare KG updates.

    Merges KG updates from all results, increments the iteration counter,
    and builds a human-readable Markdown summary.

    Args:
        state: Current workflow state after all agents have run.

    Returns:
        Updated workflow state with aggregated results.
    """
    all_kg_updates: list[dict[str, Any]] = []
    for result in state["results"]:
        all_kg_updates.extend(result.kg_updates)

    state["iteration"] += 1

    # Build Markdown summary
    state["kg_context"] = _format_final_output(state["results"])

    logger.info(
        "Aggregated %d results with %d KG updates (iteration %d)",
        len(state["results"]),
        len(all_kg_updates),
        state["iteration"],
    )
    return state


def _format_final_output(results: list[AgentResult]) -> str:
    """Build a Markdown summary from all agent results.

    Args:
        results: All agent results accumulated during the workflow.

    Returns:
        Formatted Markdown string.
    """
    parts = ["# PHAROS Results\n"]

    # Group by agent
    seen_agents: set[str] = set()
    for result in results:
        agent = result.agent_name
        if agent.lower() in ("router", "sentinel"):
            continue
        if agent in seen_agents:
            continue
        seen_agents.add(agent)

        parts.append(f"\n## {agent} (confidence: {result.confidence:.0%})\n\n")
        parts.append(result.content)
        parts.append("\n")

    # Sentinel summary
    sentinel_results = [r for r in results if r.agent_name.lower() == "sentinel"]
    if sentinel_results:
        last = sentinel_results[-1]
        parts.append(f"\n---\n\n{last.content}\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

_VERIFIED_AGENTS = ["oracle", "scribe", "cartographer", "alchemist", "architect"]
_FAST_AGENTS = ["general"]
_SPECIALIST_AGENTS = _VERIFIED_AGENTS + _FAST_AGENTS


def build_workflow(
    agents: dict[str, Any],
) -> StateGraph:  # type: ignore[type-arg]
    """Build and compile the PHAROS LangGraph workflow.

    The graph follows this flow::

        Router → Specialist → Sentinel → (retry or aggregate) → END

    If Sentinel scores < 0.5 and retries remain, the specialist is
    re-invoked once with the verification feedback in state.

    Args:
        agents: Dict mapping agent names to BaseAgent instances.
            Required keys: router, oracle, scribe, cartographer,
            alchemist, architect, sentinel.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    _AGENT_REGISTRY.clear()
    _AGENT_REGISTRY.update(agents)

    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("route", _make_agent_node("router"))
    for name in _SPECIALIST_AGENTS:
        graph.add_node(name, _make_agent_node(name))
    graph.add_node("sentinel", _make_agent_node("sentinel"))
    graph.add_node("aggregate", _aggregate)

    # Entry point
    graph.set_entry_point("route")

    # Conditional edges from router to specialist agents
    graph.add_conditional_edges(
        "route",
        _route_task,
        {name: name for name in _SPECIALIST_AGENTS} | {"sentinel": "sentinel"},
    )

    # Verified agents go through sentinel; fast agents skip to aggregate
    for name in _VERIFIED_AGENTS:
        graph.add_edge(name, "sentinel")
    for name in _FAST_AGENTS:
        graph.add_edge(name, "aggregate")

    # Sentinel conditionally retries or aggregates
    graph.add_conditional_edges(
        "sentinel",
        _sentinel_post_check,
        {name: name for name in _SPECIALIST_AGENTS} | {"aggregate": "aggregate"},
    )

    # Aggregate ends the workflow
    graph.add_edge("aggregate", END)

    return graph.compile()
