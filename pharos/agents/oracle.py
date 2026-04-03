"""Agent 1 — Oracle: biomedical forecasting and trend prediction.

Analyses publication trends, embedding trajectories, and knowledge-graph
topology to generate research hypotheses and forecasting reports.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pharos.agents.base import BaseAgent
from pharos.forecasting.hypothesis_gen import Hypothesis, HypothesisGenerator
from pharos.forecasting.trend_detector import EntityTrend, TrendDetector
from pharos.graph.entity_extractor import _extract_json
from pharos.orchestration.prompts import PROMPTS
from pharos.orchestration.task_models import AgentResult, Task, WorkflowState
from pharos.tools.pubmed_tools import PubMedClient

logger = logging.getLogger(__name__)


class OracleAgent(BaseAgent):
    """Biomedical forecasting agent.

    Orchestrates a multi-step pipeline:
    1. Extract key entities from the query.
    2. Compute publication trends for each entity.
    3. Generate hypotheses from trends + KG topology.
    4. Produce a narrative forecasting report.

    The ``pubmed``, ``trend_detector``, and ``hypothesis_gen`` dependencies
    can be injected for testing.
    """

    def __init__(
        self,
        ollama: Any,
        kg: Any,
        config: Any,
        *,
        pubmed: PubMedClient | None = None,
        trend_detector: TrendDetector | None = None,
        hypothesis_gen: HypothesisGenerator | None = None,
    ) -> None:
        super().__init__(ollama, kg, config)
        self.pubmed = pubmed or PubMedClient(config)
        self.trend_detector = trend_detector or TrendDetector(self.pubmed, ollama, config)
        self.hypothesis_gen = hypothesis_gen or HypothesisGenerator(
            ollama, kg, self.trend_detector, config
        )

    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Execute the forecasting pipeline.

        Args:
            task: Task whose ``query`` describes the forecast request.
            state: Current workflow state.

        Returns:
            AgentResult with narrative report, hypotheses, and trends.
        """
        task_id = str(uuid.uuid4())
        query = task.query

        # 1. Extract key entities from query
        entities = await self._extract_entities(query)
        if not entities:
            entities = [query]  # fallback: use the whole query

        # 2. Compute trends for each entity
        trends: list[EntityTrend] = []
        for entity in entities[:5]:  # limit to 5 entities
            trend = await self.trend_detector.compute_entity_trends(entity)
            trends.append(trend)

        # 3. Generate hypotheses
        hypotheses = await self.hypothesis_gen.generate_hypotheses(
            topic=query,
            n_hypotheses=5,
            entity_trends=trends,
        )

        # 4. Generate narrative report
        report = await self._generate_report(query, trends, hypotheses)

        # 5. Compute confidence
        if hypotheses:
            avg_confidence = sum(h.confidence for h in hypotheses) / len(hypotheses)
        else:
            avg_confidence = 0.2

        return AgentResult(
            agent_name="Oracle",
            task_id=task_id,
            content=report,
            artifacts={
                "hypotheses": [h.model_dump() for h in hypotheses],
                "trends": [t.model_dump() for t in trends],
                "entities_analysed": entities[:5],
            },
            confidence=min(0.95, avg_confidence),
        )

    # ------------------------------------------------------------------
    # Step 1: Entity extraction
    # ------------------------------------------------------------------

    async def _extract_entities(self, query: str) -> list[str]:
        """Extract key biomedical entities from the forecast query.

        Args:
            query: User's forecasting query.

        Returns:
            List of entity name strings.
        """
        try:
            raw = await self.ollama.generate(
                model=self.config.model_extractor,
                prompt=f"Extract key biomedical entities from: {query}",
                system=PROMPTS["oracle_entities"],
                format="json",
            )
            data = _extract_json(raw)
            if isinstance(data, list):
                return [str(e) for e in data if isinstance(e, str) and e.strip()]
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return [str(e) for e in v if isinstance(e, str) and e.strip()]
        except Exception:
            logger.warning("Entity extraction from query failed")
        return []

    # ------------------------------------------------------------------
    # Step 4: Report generation
    # ------------------------------------------------------------------

    async def _generate_report(
        self,
        query: str,
        trends: list[EntityTrend],
        hypotheses: list[Hypothesis],
    ) -> str:
        """Generate a narrative forecasting report.

        Args:
            query: Original user query.
            trends: Computed entity trends.
            hypotheses: Generated hypotheses.

        Returns:
            Markdown-formatted report.
        """
        prompt_parts = [f"Forecasting query: {query}\n\n"]

        if trends:
            prompt_parts.append("## Trend Data\n")
            for trend in trends:
                prompt_parts.append(
                    f"- **{trend.entity}**: velocity={trend.velocity:.1f} pub/yr, "
                    f"acceleration={trend.acceleration:.1f}, "
                    f"semantic_drift={trend.semantic_drift:.3f}\n"
                )
                if trend.yearly_counts:
                    recent = dict(sorted(trend.yearly_counts.items())[-5:])
                    prompt_parts.append(f"  Recent counts: {recent}\n")
                if trend.emerging_associations:
                    top = ", ".join(f"{n} ({s:.2f})" for n, s in trend.emerging_associations[:5])
                    prompt_parts.append(f"  Emerging associations: {top}\n")

        if hypotheses:
            prompt_parts.append("\n## Generated Hypotheses\n")
            for i, h in enumerate(hypotheses, 1):
                prompt_parts.append(
                    f"{i}. [{h.confidence:.0%} confidence, {h.time_horizon}] {h.statement}\n"
                )
                if h.evidence:
                    for ev in h.evidence:
                        prompt_parts.append(f"   - {ev}\n")

        prompt = "".join(prompt_parts)

        return await self.ollama.generate(
            model=self.config.model_reasoner,
            prompt=prompt,
            system=PROMPTS["oracle_report"],
        )
