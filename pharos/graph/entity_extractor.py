"""Biomedical entity and relation extraction from unstructured text via LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pharos.config import Settings
    from pharos.tools.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

ENTITY_TYPES = ["Gene", "Protein", "Disease", "Drug", "Pathway", "CellType", "Organism"]

_VALID_RELATION_TYPES = {
    "TARGETS",
    "TREATS",
    "CAUSES",
    "INHIBITS",
    "ACTIVATES",
    "ASSOCIATED_WITH",
    "EXPRESSED_IN",
    "PART_OF",
    "INTERACTS_WITH",
}

_ENTITY_SYSTEM_PROMPT = """\
You are a biomedical named entity recognition (NER) system.
Extract all biomedical entities from the provided text.

Entity types: {entity_types}

Return a JSON array of objects, each with:
- "name": canonical entity name
- "entity_type": one of the entity types above
- "aliases": list of alternative names found in the text (may be empty)
- "identifiers": dict of known IDs, e.g. {{"uniprot": "P04637"}} (may be empty)

## Examples

Input: "TP53 mutations are frequently found in non-small cell lung cancer (NSCLC) \
and are associated with poor response to cisplatin chemotherapy."

Output:
```json
[
  {{"name": "TP53", "entity_type": "Gene", "aliases": ["p53", "tumor protein p53"], "identifiers": {{}}}},
  {{"name": "non-small cell lung cancer", "entity_type": "Disease", "aliases": ["NSCLC"], "identifiers": {{}}}},
  {{"name": "cisplatin", "entity_type": "Drug", "aliases": [], "identifiers": {{}}}}
]
```

Input: "The MAPK/ERK signaling pathway is activated by EGFR in A549 cells, \
leading to increased proliferation."

Output:
```json
[
  {{"name": "MAPK/ERK signaling pathway", "entity_type": "Pathway", "aliases": ["MAPK pathway", "ERK pathway"], "identifiers": {{}}}},
  {{"name": "EGFR", "entity_type": "Protein", "aliases": ["epidermal growth factor receptor"], "identifiers": {{}}}},
  {{"name": "A549", "entity_type": "CellType", "aliases": [], "identifiers": {{}}}}
]
```

Return ONLY a valid JSON array. No explanation, no markdown."""

_RELATION_SYSTEM_PROMPT = """\
You are a biomedical relation extraction system.
Given a text and a list of entities already identified, extract relations between them.

Valid relation types: {relation_types}

Return a JSON array of objects, each with:
- "source": source entity name (must match one of the provided entities)
- "target": target entity name (must match one of the provided entities)
- "relation_type": one of the valid relation types above
- "confidence": float between 0.0 and 1.0
- "evidence": the exact sentence or phrase from the text supporting this relation

## Examples

Entities: ["TP53", "non-small cell lung cancer", "cisplatin"]
Text: "TP53 mutations are frequently found in non-small cell lung cancer (NSCLC) \
and are associated with poor response to cisplatin chemotherapy."

Output:
```json
[
  {{"source": "TP53", "target": "non-small cell lung cancer", "relation_type": "ASSOCIATED_WITH", \
"confidence": 0.9, "evidence": "TP53 mutations are frequently found in non-small cell lung cancer"}},
  {{"source": "cisplatin", "target": "non-small cell lung cancer", "relation_type": "TREATS", \
"confidence": 0.7, "evidence": "poor response to cisplatin chemotherapy"}}
]
```

Return ONLY a valid JSON array. No explanation, no markdown."""

_SIMPLIFIED_ENTITY_PROMPT = """\
Extract biomedical entities from this text.
Return a JSON array: [{{"name": "...", "entity_type": "Gene|Protein|Disease|Drug|Pathway|CellType|Organism"}}]
Only JSON, no other text."""


class BioEntity(BaseModel):
    """A biomedical entity extracted from text.

    Attributes:
        name: Canonical entity name (normalized).
        entity_type: Category from ENTITY_TYPES.
        aliases: Alternative names found in text.
        identifiers: Known database identifiers.
    """

    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)


class BioRelation(BaseModel):
    """A relation between two biomedical entities.

    Attributes:
        source: Source entity name.
        target: Target entity name.
        relation_type: Relation category.
        confidence: Extraction confidence (0.0–1.0).
        evidence: Supporting text fragment.
    """

    source: str
    target: str
    relation_type: str
    confidence: float = 0.0
    evidence: str = ""


class BioEntityExtractor:
    """LLM-based biomedical entity and relation extractor.

    Uses the MODEL_EXTRACTOR to extract structured entities and relations
    from unstructured biomedical text via JSON-mode prompting.

    Args:
        ollama: Async Ollama API client.
        config: Application settings.
    """

    ENTITY_TYPES = ENTITY_TYPES

    def __init__(self, ollama: OllamaClient, config: Settings) -> None:
        self._ollama = ollama
        self._model = config.model_extractor
        self._max_retries = config.max_retries

    async def extract_entities(self, text: str) -> list[BioEntity]:
        """Extract biomedical entities from text using the LLM.

        Sends a structured prompt with one-shot examples. Falls back to a
        simplified prompt if the first attempt produces unparseable output.

        Args:
            text: Biomedical text to analyse.

        Returns:
            List of extracted BioEntity objects.
        """
        system = _ENTITY_SYSTEM_PROMPT.format(entity_types=", ".join(ENTITY_TYPES))

        try:
            raw = await self._ollama.generate(
                model=self._model,
                prompt=f"Extract entities from:\n\n{text}",
                system=system,
                format="json",
            )
            entities = self._parse_entities(raw)
            if entities:
                return entities
        except Exception:
            logger.warning("Entity extraction first attempt failed")

        # Fallback: simplified prompt
        try:
            raw = await self._ollama.generate(
                model=self._model,
                prompt=f"{_SIMPLIFIED_ENTITY_PROMPT}\n\nText: {text}",
                format="json",
            )
            return self._parse_entities(raw)
        except Exception:
            logger.error("Entity extraction fallback also failed")
            return []

    async def extract_relations(self, text: str, entities: list[BioEntity]) -> list[BioRelation]:
        """Extract relations between known entities from text.

        Args:
            text: Biomedical text to analyse.
            entities: Previously extracted entities to link.

        Returns:
            List of extracted BioRelation objects.
        """
        if len(entities) < 2:
            return []

        entity_names = [e.name for e in entities]
        system = _RELATION_SYSTEM_PROMPT.format(
            relation_types=", ".join(sorted(_VALID_RELATION_TYPES)),
        )
        prompt = f"Entities: {json.dumps(entity_names)}\n\nText: {text}"

        try:
            raw = await self._ollama.generate(
                model=self._model,
                prompt=prompt,
                system=system,
                format="json",
            )
            return self._parse_relations(raw, entity_names)
        except Exception:
            logger.error("Relation extraction failed")
            return []

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_entities(self, raw: str) -> list[BioEntity]:
        """Parse LLM output into BioEntity objects.

        Handles markdown code blocks and tolerates minor JSON issues.

        Args:
            raw: Raw LLM output string.

        Returns:
            List of validated BioEntity objects.
        """
        data = _extract_json(raw)
        if data is None:
            return []

        if isinstance(data, dict):
            # Some models wrap the array in a key
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                return []

        if not isinstance(data, list):
            return []

        entities: list[BioEntity] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = _normalize_name(item.get("name", ""))
            etype = item.get("entity_type", "")
            if not name or etype not in ENTITY_TYPES:
                continue
            aliases = [_normalize_name(a) for a in item.get("aliases", []) if isinstance(a, str)]
            identifiers = item.get("identifiers", {})
            if not isinstance(identifiers, dict):
                identifiers = {}
            entities.append(
                BioEntity(
                    name=name,
                    entity_type=etype,
                    aliases=aliases,
                    identifiers={k: str(v) for k, v in identifiers.items()},
                )
            )
        return entities

    def _parse_relations(self, raw: str, entity_names: list[str]) -> list[BioRelation]:
        """Parse LLM output into BioRelation objects.

        Args:
            raw: Raw LLM output string.
            entity_names: Valid entity names for filtering.

        Returns:
            List of validated BioRelation objects.
        """
        data = _extract_json(raw)
        if data is None:
            return []

        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                return []

        if not isinstance(data, list):
            return []

        name_set = {_normalize_name(n) for n in entity_names}
        relations: list[BioRelation] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            source = _normalize_name(item.get("source", ""))
            target = _normalize_name(item.get("target", ""))
            rel_type = str(item.get("relation_type", "")).upper()
            if not source or not target or source == target:
                continue
            if rel_type not in _VALID_RELATION_TYPES:
                continue
            if source not in name_set or target not in name_set:
                continue
            confidence = item.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            confidence = max(0.0, min(1.0, float(confidence)))
            evidence = str(item.get("evidence", ""))
            relations.append(
                BioRelation(
                    source=source,
                    target=target,
                    relation_type=rel_type,
                    confidence=confidence,
                    evidence=evidence,
                )
            )
        return relations


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _extract_json(raw: str) -> Any:
    """Extract and parse JSON from LLM output.

    Handles raw JSON, markdown code fences, and embedded JSON in text.

    Args:
        raw: Raw string that may contain JSON.

    Returns:
        Parsed Python object, or None if parsing fails.
    """
    text = raw.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array or object in the text
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    return None


def _normalize_name(name: str) -> str:
    """Normalize an entity name: strip whitespace and lowercase.

    Args:
        name: Raw entity name.

    Returns:
        Normalized name string.
    """
    return name.strip().lower()
