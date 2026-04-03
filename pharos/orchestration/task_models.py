"""Pydantic models for tasks, agent results, and workflow state."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    """Supported task types that the Router can classify."""

    FORECAST = "forecast"
    REVIEW = "review"
    BUILD_KG = "build_kg"
    DESIGN_MOLECULE = "design_molecule"
    DESIGN_PROTEIN = "design_protein"
    VERIFY = "verify"
    GENERAL = "general"


class Task(BaseModel):
    """A research task submitted by the user or decomposed by the Router.

    Attributes:
        query: Natural-language research question or instruction.
        task_type: Classified task type; None means the Router will decide.
        context: Arbitrary context dict (prior results, user prefs, etc.).
        parent_task_id: ID of the parent task if this is a sub-task.
    """

    query: str
    task_type: TaskType | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    parent_task_id: str | None = None


class AgentResult(BaseModel):
    """Output produced by a single agent execution.

    Attributes:
        agent_name: Name of the agent that produced this result.
        task_id: Unique identifier for the task execution.
        content: Natural-language output text.
        artifacts: Structured data (molecules, sequences, graphs, etc.).
        confidence: Self-assessed confidence score (0.0–1.0).
        kg_updates: List of KG triple dicts to persist.
    """

    agent_name: str
    task_id: str
    content: str
    artifacts: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    kg_updates: list[dict[str, Any]] = Field(default_factory=list)


class WorkflowState(TypedDict):
    """Shared state passed through the LangGraph workflow.

    Attributes:
        task: The current research task.
        results: Accumulated agent results.
        current_agent: Name of the currently executing agent.
        kg_context: Relevant KG context retrieved for this task.
        iteration: Current workflow iteration count.
    """

    task: Task
    results: list[AgentResult]
    current_agent: str
    kg_context: str
    iteration: int
