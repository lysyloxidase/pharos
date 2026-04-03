"""Agent — General: answers simple biomedical questions with a single LLM call.

Used for queries that don't require forecasting, review, KG building, or
molecule/protein design. Fast path — one LLM call, no PubMed, no embeddings.
"""

from __future__ import annotations

import uuid
from typing import Any

from pharos.agents.base import BaseAgent
from pharos.orchestration.task_models import AgentResult, Task, WorkflowState

_SYSTEM_PROMPT = """\
You are PHAROS, a biomedical research assistant. Answer the user's question
clearly and concisely using your scientific knowledge. Include relevant details
about mechanisms, pathways, and clinical significance where appropriate.
Use Markdown formatting for readability.
"""


class GeneralAgent(BaseAgent):
    """Answers general biomedical questions with a single LLM call."""

    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Answer a general biomedical question.

        Args:
            task: The user's question.
            state: Current workflow state.

        Returns:
            AgentResult with the answer.
        """
        response = await self.ollama.generate(
            model=self.config.model_extractor,
            prompt=task.query,
            system=_SYSTEM_PROMPT,
        )

        return AgentResult(
            agent_name="General",
            task_id=str(uuid.uuid4()),
            content=response,
            confidence=0.75,
        )
