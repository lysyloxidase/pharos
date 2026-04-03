"""Session state management for LangGraph workflows.

Provides helpers to initialize, snapshot, and restore workflow state
across LangGraph invocations within a single user session.
"""

from __future__ import annotations

from typing import Any

from pharos.orchestration.task_models import AgentResult, Task, WorkflowState


def create_initial_state(task: Task) -> WorkflowState:
    """Create a fresh WorkflowState for a new task.

    Args:
        task: The incoming research task.

    Returns:
        Initialized WorkflowState dict.
    """
    return WorkflowState(
        task=task,
        results=[],
        current_agent="router",
        kg_context="",
        iteration=0,
    )


def add_result(state: WorkflowState, result: AgentResult) -> WorkflowState:
    """Append an agent result to the workflow state.

    Args:
        state: Current workflow state.
        result: Agent result to add.

    Returns:
        Updated workflow state with the new result appended.
    """
    state["results"].append(result)
    return state


def get_latest_result(state: WorkflowState) -> AgentResult | None:
    """Get the most recent agent result from state.

    Args:
        state: Current workflow state.

    Returns:
        The last AgentResult, or None if no results exist.
    """
    return state["results"][-1] if state["results"] else None


def serialize_state(state: WorkflowState) -> dict[str, Any]:
    """Serialize workflow state for persistence or logging.

    Args:
        state: Current workflow state.

    Returns:
        JSON-serializable dict.
    """
    return {
        "task": state["task"].model_dump(),
        "results": [r.model_dump() for r in state["results"]],
        "current_agent": state["current_agent"],
        "kg_context": state["kg_context"],
        "iteration": state["iteration"],
    }
