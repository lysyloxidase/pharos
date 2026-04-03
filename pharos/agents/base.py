"""Abstract base class for all PHAROS agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pharos.config import Settings
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.orchestration.task_models import AgentResult, Task, WorkflowState
    from pharos.tools.ollama_client import OllamaClient


class BaseAgent(ABC):
    """Base class that every PHAROS agent must extend.

    Provides shared access to the Ollama LLM client, Neo4j knowledge-graph
    manager, and application configuration.

    Args:
        ollama: Async Ollama API wrapper.
        kg: Neo4j knowledge-graph manager.
        config: Application settings.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        kg: Neo4jManager,
        config: Settings,
    ) -> None:
        self.ollama = ollama
        self.kg = kg
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Execute the agent's main logic for the given task.

        Args:
            task: The research task to process.
            state: Current workflow state shared across agents.

        Returns:
            An AgentResult with the agent's output, confidence score,
            and any knowledge-graph updates.
        """

    async def query_kg(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a Cypher query against the knowledge graph.

        Args:
            cypher: Cypher query string.
            params: Optional query parameters.

        Returns:
            List of result records as dictionaries.
        """
        return await self.kg.query(cypher, params or {})

    async def update_kg(self, triples: list[dict[str, Any]]) -> None:
        """Persist a list of triples to the knowledge graph.

        Each triple dict must contain 'source', 'relation', and 'target' keys.

        Args:
            triples: Knowledge-graph triples to add.
        """
        for triple in triples:
            source = triple["source"]
            target = triple["target"]
            relation = triple["relation"]
            source_id = await self.kg.add_node(
                source.get("type", "Entity"),
                source.get("properties", {}),
            )
            target_id = await self.kg.add_node(
                target.get("type", "Entity"),
                target.get("properties", {}),
            )
            await self.kg.add_relation(
                source_id,
                target_id,
                relation,
                triple.get("properties", {}),
            )
