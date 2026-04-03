"""Agent 3 — Cartographer: knowledge-graph builder.

Searches PubMed for relevant articles, extracts biomedical entities and
relations via LLM, deduplicates them, and persists everything into the
Neo4j knowledge graph.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pharos.agents.base import BaseAgent
from pharos.graph.entity_extractor import BioEntity, BioEntityExtractor, BioRelation
from pharos.orchestration.task_models import AgentResult, Task, WorkflowState
from pharos.tools.pubmed_tools import PubMedClient

logger = logging.getLogger(__name__)

_ENTITY_TYPE_TO_NODE_TYPE: dict[str, str] = {
    "gene": "Gene",
    "protein": "Protein",
    "disease": "Disease",
    "drug": "Drug",
    "pathway": "Pathway",
    "compound": "Compound",
    "celltype": "Phenotype",
    "organism": "Anatomy",
}


class CartographerAgent(BaseAgent):
    """Knowledge-graph building agent.

    Orchestrates the full KG-building pipeline:
    1. Search PubMed for articles matching the query.
    2. Fetch abstracts for the top results.
    3. Extract entities and relations from each abstract.
    4. Deduplicate entities via fuzzy name matching.
    5. Persist nodes and relations into Neo4j.

    The ``pubmed`` and ``extractor`` dependencies can be injected for testing.
    """

    def __init__(
        self,
        ollama: Any,
        kg: Any,
        config: Any,
        *,
        pubmed: PubMedClient | None = None,
        extractor: BioEntityExtractor | None = None,
    ) -> None:
        super().__init__(ollama, kg, config)
        self.pubmed = pubmed or PubMedClient(config)
        self.extractor = extractor or BioEntityExtractor(ollama, config)

    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Build or extend the knowledge graph from the given task.

        Args:
            task: The KG-building task — ``query`` describes what to map.
            state: Current workflow state.

        Returns:
            AgentResult with summary and artifact counts.
        """
        task_id = str(uuid.uuid4())
        query = task.query

        # 1. Search PubMed
        pmids = await self.pubmed.search(query, max_results=20)
        if not pmids:
            return AgentResult(
                agent_name="Cartographer",
                task_id=task_id,
                content="No PubMed articles found for the query.",
                confidence=0.1,
            )

        # 2. Fetch abstracts
        articles = await self.pubmed.fetch_abstracts(pmids)
        texts = [a.abstract for a in articles if a.abstract]

        # 3. Extract entities and relations from each abstract
        all_entities: list[BioEntity] = []
        all_relations: list[BioRelation] = []

        for text in texts:
            entities = await self.extractor.extract_entities(text)
            relations = await self.extractor.extract_relations(text, entities)
            all_entities.extend(entities)
            all_relations.extend(relations)

        # 4. Deduplicate entities
        deduped = _deduplicate_entities(all_entities)

        # 5. Persist to KG
        nodes_added = 0
        relations_added = 0
        kg_updates: list[dict[str, Any]] = []

        for entity in deduped:
            node_type = _map_entity_type(entity.entity_type)
            props: dict[str, Any] = {"name": entity.name}
            if entity.aliases:
                props["aliases"] = ", ".join(entity.aliases)
            for k, v in entity.identifiers.items():
                props[k] = v
            await self.kg.add_node(node_type, props)
            nodes_added += 1

        name_to_type = {e.name: e.entity_type for e in deduped}

        for rel in all_relations:
            source_norm = rel.source
            target_norm = rel.target
            if source_norm not in name_to_type or target_norm not in name_to_type:
                continue
            source_node_type = _map_entity_type(name_to_type[source_norm])
            target_node_type = _map_entity_type(name_to_type[target_norm])
            await self.kg.add_relation(
                source_norm,
                target_norm,
                rel.relation_type,
                {"confidence": str(rel.confidence), "evidence": rel.evidence},
            )
            relations_added += 1
            kg_updates.append(
                {
                    "source": {"type": source_node_type, "properties": {"name": source_norm}},
                    "relation": rel.relation_type,
                    "target": {"type": target_node_type, "properties": {"name": target_norm}},
                    "properties": {"confidence": str(rel.confidence)},
                }
            )

        # 6. Build top entities summary
        entity_counts: dict[str, int] = {}
        for e in all_entities:
            entity_counts[e.name] = entity_counts.get(e.name, 0) + 1
        top_entities = sorted(entity_counts, key=entity_counts.get, reverse=True)[:10]  # type: ignore[arg-type]

        content = (
            f"Knowledge graph updated from {len(texts)} abstracts. "
            f"Added {nodes_added} nodes and {relations_added} relations. "
            f"Top entities: {', '.join(top_entities)}."
        )

        return AgentResult(
            agent_name="Cartographer",
            task_id=task_id,
            content=content,
            artifacts={
                "nodes_added": nodes_added,
                "relations_added": relations_added,
                "top_entities": top_entities,
                "articles_processed": len(texts),
            },
            confidence=min(0.9, 0.3 + 0.03 * len(texts)),
            kg_updates=kg_updates,
        )


def _map_entity_type(entity_type: str) -> str:
    """Map extractor entity types to Neo4j node labels.

    Args:
        entity_type: Raw entity type string from the extractor.

    Returns:
        Neo4j node label.
    """
    return _ENTITY_TYPE_TO_NODE_TYPE.get(entity_type.lower(), "Entity")


def _deduplicate_entities(entities: list[BioEntity]) -> list[BioEntity]:
    """Deduplicate entities by normalized name and type.

    When duplicates are found, aliases and identifiers are merged.

    Args:
        entities: List of potentially duplicate entities.

    Returns:
        Deduplicated entity list.
    """
    seen: dict[tuple[str, str], BioEntity] = {}
    for entity in entities:
        key = (entity.name, entity.entity_type.lower())
        if key in seen:
            existing = seen[key]
            merged_aliases = list(set(existing.aliases) | set(entity.aliases))
            merged_ids = {**existing.identifiers, **entity.identifiers}
            seen[key] = BioEntity(
                name=entity.name,
                entity_type=entity.entity_type,
                aliases=merged_aliases,
                identifiers=merged_ids,
            )
        else:
            seen[key] = entity
    return list(seen.values())
