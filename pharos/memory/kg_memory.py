"""KG-based long-term memory for agents.

Stores and retrieves agent observations, reasoning traces, and learned facts
as nodes/relations in the knowledge graph so they persist across sessions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.tools.ollama_client import OllamaClient


class KGMemory:
    """Long-term memory backed by the Neo4j knowledge graph.

    Agents can store observations and retrieve relevant context from
    past interactions via semantic search over the KG.

    Args:
        kg: Neo4j manager instance.
        ollama: Ollama client for embedding generation.
        embedding_model: Name of the embedding model.
    """

    def __init__(
        self,
        kg: Neo4jManager,
        ollama: OllamaClient,
        embedding_model: str = "all-minilm:l6-v2",
    ) -> None:
        self._kg = kg
        self._ollama = ollama
        self._embedding_model = embedding_model

    async def store(
        self,
        agent_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store an observation in long-term memory.

        Args:
            agent_name: Name of the agent storing the memory.
            content: Text content to remember.
            metadata: Optional metadata dict.

        Returns:
            The node name (ID) of the stored memory.
        """
        import hashlib
        import time

        mem_id = hashlib.sha256(f"{agent_name}:{content}:{time.time()}".encode()).hexdigest()[:16]
        name = f"memory_{mem_id}"

        props: dict[str, Any] = {
            "name": name,
            "agent": agent_name,
            "content": content,
            "timestamp": time.time(),
        }
        if metadata:
            props.update(metadata)

        await self._kg.add_node("Memory", props)
        return name

    async def recall(
        self,
        query: str,
        agent_name: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories via fulltext search.

        Args:
            query: Search query.
            agent_name: Optional filter by agent name.
            limit: Maximum results.

        Returns:
            List of memory dicts.
        """
        cypher = (
            "CALL db.index.fulltext.queryNodes('entity_fulltext', $query) "
            "YIELD node, score "
            "WHERE 'Memory' IN labels(node) "
        )
        if agent_name:
            cypher += "AND node.agent = $agent "
        cypher += "RETURN node {.*} AS memory, score ORDER BY score DESC LIMIT $limit"

        params: dict[str, Any] = {"query": query, "limit": limit}
        if agent_name:
            params["agent"] = agent_name

        return await self._kg.query(cypher, params)
