"""Agent 0 — Router: classifies incoming queries and routes to specialist agents.

The Router is the entry-point of every PHAROS workflow.  It uses a small,
fast LLM (MODEL_ROUTER) to determine the TaskType and optionally decompose
complex queries into sub-tasks.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from pharos.agents.base import BaseAgent
from pharos.orchestration.task_models import AgentResult, Task, TaskType, WorkflowState

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """\
You are PHAROS Router, a biomedical research task classifier.

Given a user query, respond with a JSON object containing:
- "task_type": one of {task_types}
- "sub_tasks": optional list of strings if the query should be decomposed

## Task type descriptions and examples

- **forecast**: Predict future biomedical trends, drug approvals, clinical outcomes.
  Examples: "Will CRISPR therapies for sickle cell be approved by 2026?",
  "Predict the next breakthrough target in Alzheimer's research."

- **review**: Write a narrative literature review or summarize research.
  Examples: "Write a review of CAR-T therapy advances in solid tumors.",
  "Summarize recent findings on gut-brain axis."

- **build_kg**: Build or extend a knowledge graph from literature/data.
  Examples: "Map all known interactions of TP53.",
  "Build a knowledge graph of EGFR signaling pathway."

- **design_molecule**: Design, optimize, or evaluate small molecules.
  Examples: "Design a selective JAK2 inhibitor with good oral bioavailability.",
  "Optimize this SMILES for blood-brain barrier penetration: CCO..."

- **design_protein**: Design or analyze proteins and protein structures.
  Examples: "Design a nanobody targeting PD-L1.",
  "Predict the structure of this sequence: MKTL..."

- **verify**: Fact-check a biomedical claim or verify experimental results.
  Examples: "Is it true that metformin reduces cancer risk?",
  "Verify: BRCA1 mutations cause a 70% lifetime risk of breast cancer."

- **general**: General biomedical question that doesn't fit other categories.
  Examples: "What is the mechanism of action of ibuprofen?",
  "Explain how mRNA vaccines work."

Respond ONLY with valid JSON. No markdown, no explanation.
"""

_KEYWORD_MAP: dict[str, TaskType] = {
    "predict": TaskType.FORECAST,
    "forecast": TaskType.FORECAST,
    "will": TaskType.FORECAST,
    "future": TaskType.FORECAST,
    "review": TaskType.REVIEW,
    "summarize": TaskType.REVIEW,
    "summary": TaskType.REVIEW,
    "literature": TaskType.REVIEW,
    "knowledge graph": TaskType.BUILD_KG,
    "build kg": TaskType.BUILD_KG,
    "map interactions": TaskType.BUILD_KG,
    "pathway": TaskType.BUILD_KG,
    "design molecule": TaskType.DESIGN_MOLECULE,
    "smiles": TaskType.DESIGN_MOLECULE,
    "inhibitor": TaskType.DESIGN_MOLECULE,
    "drug design": TaskType.DESIGN_MOLECULE,
    "molecular": TaskType.DESIGN_MOLECULE,
    "design protein": TaskType.DESIGN_PROTEIN,
    "nanobody": TaskType.DESIGN_PROTEIN,
    "protein structure": TaskType.DESIGN_PROTEIN,
    "protein design": TaskType.DESIGN_PROTEIN,
    "sequence": TaskType.DESIGN_PROTEIN,
    "verify": TaskType.VERIFY,
    "fact-check": TaskType.VERIFY,
    "is it true": TaskType.VERIFY,
    "check claim": TaskType.VERIFY,
    "what is": TaskType.GENERAL,
    "what are": TaskType.GENERAL,
    "how does": TaskType.GENERAL,
    "explain": TaskType.GENERAL,
    "describe": TaskType.GENERAL,
    "define": TaskType.GENERAL,
    "tell me about": TaskType.GENERAL,
}


class RouterAgent(BaseAgent):
    """Classifies user queries into TaskTypes and routes to specialist agents.

    Uses a small LLM for fast classification with a keyword-matching fallback
    when the model response cannot be parsed.
    """

    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Classify the query and determine routing.

        Args:
            task: The incoming user task.
            state: Current workflow state.

        Returns:
            AgentResult whose ``artifacts`` dict contains ``task_type`` and
            optionally ``sub_tasks``.
        """
        task_type, sub_tasks = await self._classify(task.query)

        task.task_type = task_type

        return AgentResult(
            agent_name="Router",
            task_id=str(uuid.uuid4()),
            content=f"Classified as {task_type.value}",
            artifacts={
                "task_type": task_type.value,
                "sub_tasks": sub_tasks,
            },
            confidence=0.9 if sub_tasks is None else 0.85,
        )

    async def _classify(self, query: str) -> tuple[TaskType, list[str] | None]:
        """Classify a query — fast keyword match first, LLM only if no match.

        Args:
            query: Raw user query string.

        Returns:
            Tuple of (TaskType, optional list of sub-task strings).
        """
        # Fast path: keyword match avoids an LLM call entirely
        keyword_hit, matched = self._keyword_fallback_ex(query)
        if matched:
            return keyword_hit, None

        # Slow path: no keyword matched, use LLM
        task_types_str = ", ".join(t.value for t in TaskType)
        system = ROUTER_SYSTEM_PROMPT.format(task_types=task_types_str)
        prompt = f"Classify this biomedical query:\n\n{query}"

        try:
            response = await self.ollama.generate(
                model=self.config.model_router,
                prompt=prompt,
                system=system,
                format="json",
            )
            return self._parse_llm_response(response)
        except Exception:
            logger.warning("LLM classification failed, returning general")
            return TaskType.GENERAL, None

    def _parse_llm_response(self, response: str) -> tuple[TaskType, list[str] | None]:
        """Parse the JSON response from the router LLM.

        Args:
            response: Raw LLM response string (expected JSON).

        Returns:
            Parsed (TaskType, sub_tasks) tuple.
        """
        try:
            data: dict[str, Any] = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return TaskType.GENERAL, None

        raw_type = data.get("task_type", "general")
        try:
            task_type = TaskType(raw_type)
        except ValueError:
            task_type = TaskType.GENERAL

        sub_tasks: list[str] | None = data.get("sub_tasks")
        if sub_tasks is not None and not isinstance(sub_tasks, list):
            sub_tasks = None

        return task_type, sub_tasks

    @staticmethod
    def _keyword_fallback(query: str) -> TaskType:
        """Determine task type from keyword matching.

        Args:
            query: Raw user query string.

        Returns:
            Best-matching TaskType, or GENERAL as default.
        """
        return RouterAgent._keyword_fallback_ex(query)[0]

    @staticmethod
    def _keyword_fallback_ex(query: str) -> tuple[TaskType, bool]:
        """Determine task type from keyword matching with match indicator.

        Args:
            query: Raw user query string.

        Returns:
            Tuple of (TaskType, whether a keyword actually matched).
        """
        query_lower = query.lower()
        for keyword, task_type in _KEYWORD_MAP.items():
            if keyword in query_lower:
                return task_type, True
        return TaskType.GENERAL, False
