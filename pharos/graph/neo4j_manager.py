"""Neo4j knowledge-graph manager — async connection, CRUD, schema setup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from pharos.graph.schema import NodeType, RelationType

if TYPE_CHECKING:
    from pharos.config import Settings

logger = logging.getLogger(__name__)

# Constraints and indexes applied during setup_schema()
_CONSTRAINTS: list[str] = [
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{nt.value}) REQUIRE n.name IS UNIQUE"
    for nt in NodeType
]

_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
    "FOR (n:Gene|Disease|Drug|Protein|Pathway|Compound|Phenotype|Anatomy) "
    "ON EACH [n.name, n.description]"
)


class Neo4jManager:
    """Async Neo4j driver wrapper with schema management and CRUD helpers.

    Usage::

        async with Neo4jManager(settings) as kg:
            await kg.setup_schema()
            node_id = await kg.add_node("Gene", {"name": "TP53"})

    Args:
        config: Application settings containing Neo4j connection details.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._driver: AsyncDriver | None = None

    async def __aenter__(self) -> Neo4jManager:
        """Open the Neo4j driver connection."""
        self._driver = AsyncGraphDatabase.driver(
            self._config.neo4j_uri,
            auth=(self._config.neo4j_user, self._config.neo4j_password),
        )
        logger.info("Connected to Neo4j at %s", self._config.neo4j_uri)
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4jManager is not connected. Use 'async with' context.")
        return self._driver

    def _session(self) -> AsyncSession:
        return self._get_driver().session()

    # --- Schema ----------------------------------------------------------

    async def setup_schema(self) -> None:
        """Create uniqueness constraints and fulltext indexes.

        Safe to call multiple times — uses ``IF NOT EXISTS``.
        """
        async with self._session() as session:
            for constraint in _CONSTRAINTS:
                await session.run(constraint)
            await session.run(_FULLTEXT_INDEX)
        logger.info("Schema setup complete (%d constraints)", len(_CONSTRAINTS))

    # --- CRUD ------------------------------------------------------------

    async def add_node(self, node_type: str, properties: dict[str, Any]) -> str:
        """Merge a node by name, setting additional properties.

        Args:
            node_type: Label for the node (must match a NodeType value).
            properties: Node properties; must include ``name``.

        Returns:
            The ``name`` property used as the node identifier.
        """
        name = properties.get("name", "")
        if not name:
            raise ValueError("Node properties must include 'name'")

        # Validate node type
        try:
            NodeType(node_type)
        except ValueError:
            logger.warning("Unknown node type '%s', using as-is", node_type)

        cypher = f"MERGE (n:{node_type} {{name: $name}}) SET n += $props RETURN n.name AS name"
        props = {k: v for k, v in properties.items() if k != "name"}
        async with self._session() as session:
            result = await session.run(cypher, name=name, props=props)
            record = await result.single()
            return str(record["name"]) if record else name

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or merge a relation between two nodes.

        Args:
            source_id: Name of the source node.
            target_id: Name of the target node.
            relation_type: Relation label (should match a RelationType value).
            properties: Optional relation properties.
        """
        # Validate relation type
        try:
            RelationType(relation_type)
        except ValueError:
            logger.warning("Unknown relation type '%s', using as-is", relation_type)

        props = properties or {}
        cypher = (
            "MATCH (a {name: $source}), (b {name: $target}) "
            f"MERGE (a)-[r:{relation_type}]->(b) "
            "SET r += $props"
        )
        async with self._session() as session:
            await session.run(cypher, source=source_id, target=target_id, props=props)

    async def query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute an arbitrary Cypher query and return results as dicts.

        Args:
            cypher: Cypher query string.
            params: Optional query parameters.

        Returns:
            List of result records.
        """
        async with self._session() as session:
            result = await session.run(cypher, **(params or {}))
            return [dict(record) async for record in result]

    async def get_neighbors(self, node_id: str, depth: int = 1) -> list[dict[str, Any]]:
        """Return the subgraph around a node up to a given depth.

        Args:
            node_id: Name of the center node.
            depth: Traversal depth (default 1).

        Returns:
            List of dicts with ``source``, ``relation``, and ``target`` keys.
        """
        cypher = (
            "MATCH (a {name: $name})-[r*1..$depth]-(b) "
            "UNWIND r AS rel "
            "RETURN startNode(rel).name AS source, type(rel) AS relation, "
            "endNode(rel).name AS target"
        )
        async with self._session() as session:
            result = await session.run(cypher, name=node_id, depth=depth)
            return [dict(record) async for record in result]

    async def search_nodes(
        self,
        text: str,
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fulltext search across entity names and descriptions.

        Args:
            text: Search query text.
            node_type: Optional filter by node type label.
            limit: Maximum results to return.

        Returns:
            List of matching nodes as dicts.
        """
        cypher = "CALL db.index.fulltext.queryNodes('entity_fulltext', $text) YIELD node, score "
        if node_type:
            cypher += f"WHERE '{node_type}' IN labels(node) "
        cypher += (
            "RETURN node {.*, _labels: labels(node)} AS n, score ORDER BY score DESC LIMIT $limit"
        )

        async with self._session() as session:
            result = await session.run(cypher, text=text, limit=limit)
            return [dict(record) async for record in result]

    async def stats(self) -> dict[str, int]:
        """Return node counts per label.

        Returns:
            Dict mapping label name to count.
        """
        cypher = (
            "CALL db.labels() YIELD label "
            "CALL { WITH label "
            "  MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt "
            "} "
            "RETURN label, cnt"
        )
        async with self._session() as session:
            result = await session.run(cypher)
            return {str(record["label"]): int(record["cnt"]) async for record in result}
