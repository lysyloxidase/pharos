"""Agent 4 — Alchemist: molecular designer.

The Alchemist agent designs, optimizes, and evaluates small molecules
using RDKit for cheminformatics and LLM reasoning for structure-activity
relationship analysis.

TODO (Phase 2):
    - Implement SMILES generation via LLM with RDKit validation
    - Add property prediction (logP, MW, TPSA, Lipinski)
    - Support molecular optimization with multi-objective scoring
    - Integrate docking score estimation
    - Add retrosynthetic analysis
"""

from __future__ import annotations

import uuid

from pharos.agents.base import BaseAgent
from pharos.orchestration.task_models import AgentResult, Task, WorkflowState


class AlchemistAgent(BaseAgent):
    """Molecular design agent.

    Designs and optimizes small molecules using a combination of LLM
    reasoning and cheminformatics tools (RDKit).
    """

    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Design or optimize molecules for the given task.

        Args:
            task: The molecular design task with query and context.
            state: Current workflow state.

        Returns:
            AgentResult with designed molecules in artifacts.
        """
        # TODO: Parse design constraints from query
        # TODO: Generate candidate SMILES via LLM
        # TODO: Validate with RDKit and compute properties
        # TODO: Rank candidates by multi-objective score
        return AgentResult(
            agent_name="Alchemist",
            task_id=str(uuid.uuid4()),
            content="[Alchemist] not yet implemented",
            confidence=0.0,
        )
