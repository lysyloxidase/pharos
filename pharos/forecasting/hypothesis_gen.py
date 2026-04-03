"""Hypothesis generation by combining publication trends with KG topology."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from pharos.forecasting.trend_detector import EntityTrend  # noqa: TC001
from pharos.graph.entity_extractor import _extract_json
from pharos.orchestration.prompts import PROMPTS

if TYPE_CHECKING:
    from pharos.config import Settings
    from pharos.forecasting.trend_detector import TrendDetector
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.tools.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


class Hypothesis(BaseModel):
    """A generated research hypothesis with supporting evidence.

    Attributes:
        statement: Natural-language hypothesis statement.
        target_entity: Proposed therapeutic target.
        disease_entity: Target disease.
        evidence: Supporting evidence strings.
        confidence: Estimated plausibility (0-1).
        time_horizon: Predicted timeline (e.g. "1-2 years").
        supporting_trends: Entity trends backing the hypothesis.
        kg_path: KG path connecting target to disease, if found.
    """

    statement: str
    target_entity: str = ""
    disease_entity: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    time_horizon: str = ""
    supporting_trends: list[EntityTrend] = Field(default_factory=list)
    kg_path: list[str] | None = None


# ------------------------------------------------------------------
# HypothesisGenerator
# ------------------------------------------------------------------


class HypothesisGenerator:
    """Generate research hypotheses from trends and knowledge-graph topology.

    Combines TrendDetector signals with KG gap analysis and LLM reasoning
    to propose novel therapeutic hypotheses.

    Args:
        ollama: Async Ollama client.
        kg: Neo4j knowledge-graph manager.
        trend_detector: TrendDetector instance.
        config: Application settings.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        kg: Neo4jManager,
        trend_detector: TrendDetector,
        config: Settings,
    ) -> None:
        self._ollama = ollama
        self._kg = kg
        self._trends = trend_detector
        self._config = config

    async def generate_hypotheses(
        self,
        topic: str,
        n_hypotheses: int = 5,
        entity_trends: list[EntityTrend] | None = None,
    ) -> list[Hypothesis]:
        """Generate research hypotheses for a topic.

        Steps:
            1. Query KG for diseases without approved drugs.
            2. Query KG for genes/proteins with rising publication counts.
            3. Use TrendDetector to find converging entities.
            4. LLM synthesises hypotheses from the above data.

        Args:
            topic: Research topic or domain.
            n_hypotheses: Number of hypotheses to generate.
            entity_trends: Pre-computed trends (skips KG queries if given).

        Returns:
            List of ranked Hypothesis objects.
        """
        # 1 & 2. KG gap analysis
        unmet_diseases = await self._find_unmet_diseases()
        trending_targets = await self._find_trending_targets()

        # 3. Convergence signals (if we have both disease and target)
        convergence_evidence: list[str] = []
        if unmet_diseases and trending_targets and entity_trends:
            for trend in entity_trends[:3]:
                for disease in unmet_diseases[:3]:
                    signal = await self._trends.find_converging_entities(trend.entity, disease)
                    if signal.convergence_rate > 0.01:
                        convergence_evidence.append(
                            f"{signal.entity_a} is converging toward {signal.entity_b} "
                            f"(rate={signal.convergence_rate:.3f})"
                        )

        # 4. LLM hypothesis generation
        prompt = self._build_hypothesis_prompt(
            topic=topic,
            unmet_diseases=unmet_diseases,
            trending_targets=trending_targets,
            entity_trends=entity_trends or [],
            convergence_evidence=convergence_evidence,
            n_hypotheses=n_hypotheses,
        )

        raw = await self._ollama.generate(
            model=self._config.model_reasoner,
            prompt=prompt,
            system=PROMPTS["oracle_hypothesis"],
            format="json",
        )

        hypotheses = self._parse_hypotheses(raw, entity_trends or [])
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:n_hypotheses]

    # ------------------------------------------------------------------
    # KG queries
    # ------------------------------------------------------------------

    async def _find_unmet_diseases(self) -> list[str]:
        """Find diseases in the KG without linked approved drugs.

        Returns:
            List of disease names.
        """
        cypher = (
            "MATCH (d:Disease) WHERE NOT (d)<-[:TREATS]-(:Drug) RETURN d.name AS name LIMIT 20"
        )
        try:
            results = await self._kg.query(cypher)
            return [str(r["name"]) for r in results if r.get("name")]
        except Exception:
            logger.warning("Failed to query unmet diseases from KG")
            return []

    async def _find_trending_targets(self) -> list[str]:
        """Find genes/proteins in the KG that have many relations.

        Returns:
            List of target entity names.
        """
        cypher = (
            "MATCH (g) WHERE g:Gene OR g:Protein "
            "WITH g, size([(g)-[]-() | 1]) AS degree "
            "ORDER BY degree DESC LIMIT 20 "
            "RETURN g.name AS name"
        )
        try:
            results = await self._kg.query(cypher)
            return [str(r["name"]) for r in results if r.get("name")]
        except Exception:
            logger.warning("Failed to query trending targets from KG")
            return []

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_hypothesis_prompt(
        self,
        topic: str,
        unmet_diseases: list[str],
        trending_targets: list[str],
        entity_trends: list[EntityTrend],
        convergence_evidence: list[str],
        n_hypotheses: int,
    ) -> str:
        """Build the LLM prompt for hypothesis generation.

        Args:
            topic: Research topic.
            unmet_diseases: Diseases without treatments.
            trending_targets: High-connectivity KG targets.
            entity_trends: Computed trend data.
            convergence_evidence: Convergence signal descriptions.
            n_hypotheses: Number of hypotheses requested.

        Returns:
            Formatted prompt string.
        """
        parts = [f"Research topic: {topic}\n"]

        if unmet_diseases:
            parts.append(
                f"Diseases without approved treatments: {', '.join(unmet_diseases[:10])}\n"
            )

        if trending_targets:
            parts.append(f"High-connectivity targets in KG: {', '.join(trending_targets[:10])}\n")

        if entity_trends:
            parts.append("Publication trends:\n")
            for trend in entity_trends[:5]:
                parts.append(
                    f"  - {trend.entity}: velocity={trend.velocity:.1f} pub/yr, "
                    f"acceleration={trend.acceleration:.1f}, "
                    f"semantic_drift={trend.semantic_drift:.3f}\n"
                )
                if trend.emerging_associations:
                    top_assoc = ", ".join(
                        f"{name} ({score:.2f})" for name, score in trend.emerging_associations[:5]
                    )
                    parts.append(f"    Emerging associations: {top_assoc}\n")

        if convergence_evidence:
            parts.append("Convergence signals:\n")
            for ev in convergence_evidence:
                parts.append(f"  - {ev}\n")

        parts.append(f"\nGenerate exactly {n_hypotheses} hypotheses.")

        return "".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_hypotheses(
        self,
        raw: str,
        entity_trends: list[EntityTrend],
    ) -> list[Hypothesis]:
        """Parse LLM JSON response into Hypothesis objects.

        Args:
            raw: Raw LLM response.
            entity_trends: Available trends for back-reference.

        Returns:
            List of parsed hypotheses.
        """
        data = _extract_json(raw)

        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                return []

        if not isinstance(data, list):
            return []

        trend_map = {t.entity.lower(): t for t in entity_trends}
        hypotheses: list[Hypothesis] = []

        for item in data:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", ""))
            if not statement:
                continue

            target = str(item.get("target_entity", ""))
            disease = str(item.get("disease_entity", ""))
            evidence = [str(e) for e in item.get("evidence", []) if isinstance(e, str)]

            confidence = item.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))

            time_horizon = str(item.get("time_horizon", "3-5 years"))
            kg_path = item.get("kg_path")
            if kg_path is not None and not isinstance(kg_path, list):
                kg_path = None

            supporting = []
            for name in (target.lower(), disease.lower()):
                if name in trend_map:
                    supporting.append(trend_map[name])

            hypotheses.append(
                Hypothesis(
                    statement=statement,
                    target_entity=target,
                    disease_entity=disease,
                    evidence=evidence,
                    confidence=confidence,
                    time_horizon=time_horizon,
                    supporting_trends=supporting,
                    kg_path=kg_path,
                )
            )

        return hypotheses
