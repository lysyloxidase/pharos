"""Agent 5 — Architect: protein designer.

Orchestrates a multi-step protein design pipeline:

1. Parse the design brief (function, constraints, strategy).
2. Query the KG for known enzymes, stabilising mutations, homologs.
3. LLM selects a strategy: de novo, redesign, or mutagenesis.
4. Execute the chosen strategy via :class:`ProteinToolkit`.
5. Rank top candidates with ESM-2 perplexity.
6. LLM summarises results into a narrative report.
7. Persist designed proteins and annotations to the KG.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from pharos.agents.base import BaseAgent
from pharos.graph.entity_extractor import _extract_json
from pharos.orchestration.prompts import PROMPTS
from pharos.orchestration.task_models import AgentResult, Task, WorkflowState
from pharos.tools.protein_tools import (
    ProteinToolkit,
)

if TYPE_CHECKING:
    from pharos.config import Settings
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.tools.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal data structures
# ------------------------------------------------------------------


class _DesignBrief:
    """Parsed design brief (internal, not a Pydantic model)."""

    __slots__ = (
        "strategy",
        "target_function",
        "constraints",
        "sequence",
        "pdb_path",
        "mutations_of_interest",
    )

    def __init__(
        self,
        strategy: str = "de_novo",
        target_function: str = "",
        constraints: list[str] | None = None,
        sequence: str | None = None,
        pdb_path: str | None = None,
        mutations_of_interest: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.target_function = target_function
        self.constraints = constraints or []
        self.sequence = sequence
        self.pdb_path = pdb_path
        self.mutations_of_interest = mutations_of_interest or []


# ------------------------------------------------------------------
# ArchitectAgent
# ------------------------------------------------------------------


class ArchitectAgent(BaseAgent):
    """Protein design agent.

    Designs proteins via de-novo generation (ProtGPT2), inverse folding
    (ProteinMPNN), or directed mutagenesis (ESM-2 masked marginal scoring),
    then ranks candidates using ESM-2 perplexity and optionally predicts
    3-D structure with ESMFold.

    Args:
        ollama: Async Ollama client.
        kg: Neo4j knowledge-graph manager.
        config: Application settings.
        toolkit: Pre-built :class:`ProteinToolkit` (injectable for testing).
    """

    def __init__(
        self,
        ollama: OllamaClient,
        kg: Neo4jManager,
        config: Settings,
        *,
        toolkit: ProteinToolkit | None = None,
    ) -> None:
        super().__init__(ollama, kg, config)
        self.toolkit = toolkit or ProteinToolkit(
            proteinmpnn_path=config.proteinmpnn_path or None,
            device="cpu",
        )
        self._max_candidates = config.architect_max_candidates

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Execute the protein design pipeline.

        Args:
            task: Task whose ``query`` describes the design request.
            state: Current workflow state.

        Returns:
            AgentResult with narrative report, candidate sequences,
            variant effects, and optional structure predictions.
        """
        task_id = str(uuid.uuid4())
        query = task.query

        # Step 1 — parse design brief
        brief = await self._parse_brief(query)

        # Step 2 — query KG for context
        kg_context = await self._query_kg_context(brief)

        # Step 3 — LLM decides strategy
        strategy_info = await self._decide_strategy(brief, kg_context)
        brief.strategy = strategy_info.get("strategy", brief.strategy)
        if strategy_info.get("suggested_mutations"):
            brief.mutations_of_interest = strategy_info["suggested_mutations"]

        # Step 4 — execute strategy
        candidates: list[str] = []
        variant_effects: list[dict[str, Any]] = []
        mpnn_designs: list[dict[str, Any]] = []

        if brief.strategy == "de_novo":
            candidates = await self._run_de_novo(brief)
        elif brief.strategy == "redesign":
            mpnn_designs = await self._run_redesign(brief)
            candidates = [d["sequence"] for d in mpnn_designs]
        elif brief.strategy == "mutagenesis":
            variant_effects = await self._run_mutagenesis(brief)
            # Candidates are the original + beneficial mutants
            candidates = self._candidates_from_mutagenesis(brief, variant_effects)

        # Step 5 — rank by ESM-2 perplexity
        ranked = await self._rank_candidates(candidates)

        # Step 6 — generate report
        report = await self._generate_report(
            query, brief, strategy_info, ranked, variant_effects, mpnn_designs
        )

        # Step 7 — persist to KG
        kg_updates = self._build_kg_updates(brief, ranked)

        return AgentResult(
            agent_name="Architect",
            task_id=task_id,
            content=report,
            artifacts={
                "strategy": brief.strategy,
                "candidates": ranked,
                "variant_effects": variant_effects,
                "mpnn_designs": mpnn_designs,
            },
            confidence=self._compute_confidence(ranked),
            kg_updates=kg_updates,
        )

    # ------------------------------------------------------------------
    # Step 1 — parse design brief
    # ------------------------------------------------------------------

    async def _parse_brief(self, query: str) -> _DesignBrief:
        """Extract structured design parameters from the user query.

        Args:
            query: Natural-language design request.

        Returns:
            Populated :class:`_DesignBrief`.
        """
        try:
            raw = await self.ollama.generate(
                model=self.config.model_extractor,
                prompt=f"Parse this protein design request:\n{query}",
                system=PROMPTS["architect_parse_brief"],
                format="json",
            )
            data = _extract_json(raw)
            if isinstance(data, dict):
                return _DesignBrief(
                    strategy=str(data.get("strategy", "de_novo")),
                    target_function=str(data.get("target_function", query)),
                    constraints=[str(c) for c in data.get("constraints", [])],
                    sequence=data.get("sequence"),
                    pdb_path=data.get("pdb_path"),
                    mutations_of_interest=[str(m) for m in data.get("mutations_of_interest", [])],
                )
        except Exception:
            logger.warning("Failed to parse design brief, using defaults")

        return _DesignBrief(target_function=query)

    # ------------------------------------------------------------------
    # Step 2 — KG context
    # ------------------------------------------------------------------

    async def _query_kg_context(self, brief: _DesignBrief) -> str:
        """Query the KG for relevant protein engineering knowledge.

        Args:
            brief: Parsed design brief.

        Returns:
            Textual summary of KG findings.
        """
        parts: list[str] = []

        # Known enzymes with the target function
        cypher_enzymes = (
            "MATCH (p:Protein)-[:HAS_FUNCTION]->(f) "
            "WHERE toLower(f.name) CONTAINS toLower($func) "
            "RETURN p.name AS name, f.name AS function LIMIT 10"
        )
        try:
            results = await self.kg.query(cypher_enzymes, {"func": brief.target_function})
            if results:
                names = [str(r["name"]) for r in results if r.get("name")]
                parts.append(f"Known proteins with related function: {', '.join(names)}")
        except Exception:
            logger.debug("KG enzyme query failed")

        # Known stabilising mutations
        cypher_mutations = (
            "MATCH (m:Mutation)-[:STABILIZES]->(p:Protein) "
            "RETURN m.name AS mutation, p.name AS protein LIMIT 10"
        )
        try:
            results = await self.kg.query(cypher_mutations)
            if results:
                muts = [f"{r['mutation']} on {r['protein']}" for r in results if r.get("mutation")]
                parts.append(f"Known stabilising mutations: {', '.join(muts)}")
        except Exception:
            logger.debug("KG mutation query failed")

        return "\n".join(parts) if parts else "No relevant KG context found."

    # ------------------------------------------------------------------
    # Step 3 — strategy decision
    # ------------------------------------------------------------------

    async def _decide_strategy(self, brief: _DesignBrief, kg_context: str) -> dict[str, Any]:
        """Ask the LLM to confirm or override the design strategy.

        Args:
            brief: Parsed design brief.
            kg_context: KG context string.

        Returns:
            Dict with strategy, rationale, target_properties, suggested_mutations.
        """
        prompt = (
            f"Design brief:\n"
            f"  Strategy: {brief.strategy}\n"
            f"  Function: {brief.target_function}\n"
            f"  Constraints: {', '.join(brief.constraints) or 'none'}\n"
            f"  Sequence provided: {'yes' if brief.sequence else 'no'}\n"
            f"  PDB provided: {'yes' if brief.pdb_path else 'no'}\n"
            f"  Mutations of interest: {', '.join(brief.mutations_of_interest) or 'none'}\n\n"
            f"KG context:\n{kg_context}\n"
        )

        try:
            raw = await self.ollama.generate(
                model=self.config.model_reasoner,
                prompt=prompt,
                system=PROMPTS["architect_strategy"],
                format="json",
            )
            data = _extract_json(raw)
            if isinstance(data, dict):
                return {
                    "strategy": str(data.get("strategy", brief.strategy)),
                    "rationale": str(data.get("rationale", "")),
                    "target_properties": data.get("target_properties", []),
                    "suggested_mutations": [str(m) for m in data.get("suggested_mutations", [])],
                }
        except Exception:
            logger.warning("Strategy decision failed, keeping default")

        return {"strategy": brief.strategy, "rationale": "", "target_properties": []}

    # ------------------------------------------------------------------
    # Step 4a — de novo generation
    # ------------------------------------------------------------------

    async def _run_de_novo(self, brief: _DesignBrief) -> list[str]:
        """Generate candidate sequences de novo via ProtGPT2.

        Args:
            brief: Design brief (used for candidate count).

        Returns:
            List of amino-acid sequences.
        """
        try:
            sequences = await self.toolkit.generate_sequences(
                n=self._max_candidates,
                max_length=200,
                temperature=1.0,
            )
            return [s for s in sequences if self.toolkit.validate_sequence(s)]
        except Exception:
            logger.warning("De novo generation failed")
            return []

    # ------------------------------------------------------------------
    # Step 4b — redesign (ProteinMPNN)
    # ------------------------------------------------------------------

    async def _run_redesign(self, brief: _DesignBrief) -> list[dict[str, Any]]:
        """Design sequences for a given backbone via ProteinMPNN.

        Args:
            brief: Design brief with ``pdb_path`` set.

        Returns:
            List of dicts with sequence, score, recovery.
        """
        if not brief.pdb_path:
            logger.warning("Redesign requested but no PDB path provided")
            return []

        try:
            designs = await self.toolkit.design_sequence_for_structure(
                pdb_path=brief.pdb_path,
                n_designs=self._max_candidates,
            )
            return [d.model_dump() for d in designs]
        except RuntimeError as exc:
            logger.warning("ProteinMPNN not available: %s", exc)
            return []
        except Exception:
            logger.warning("Redesign failed")
            return []

    # ------------------------------------------------------------------
    # Step 4c — mutagenesis
    # ------------------------------------------------------------------

    async def _run_mutagenesis(self, brief: _DesignBrief) -> list[dict[str, Any]]:
        """Score mutations on the wild-type sequence via ESM-2.

        Args:
            brief: Design brief with ``sequence`` and ``mutations_of_interest``.

        Returns:
            List of VariantEffect dicts.
        """
        if not brief.sequence:
            logger.warning("Mutagenesis requested but no WT sequence provided")
            return []

        mutations = brief.mutations_of_interest
        if not mutations:
            logger.warning("No mutations specified for mutagenesis strategy")
            return []

        try:
            effects = await self.toolkit.predict_variant_effects(brief.sequence, mutations)
            return [e.model_dump() for e in effects]
        except Exception:
            logger.warning("Variant effect prediction failed")
            return []

    @staticmethod
    def _candidates_from_mutagenesis(
        brief: _DesignBrief, variant_effects: list[dict[str, Any]]
    ) -> list[str]:
        """Build candidate sequences by applying beneficial mutations.

        Args:
            brief: Design brief with wild-type sequence.
            variant_effects: Scored variant effects.

        Returns:
            List of mutated sequences (including wild-type).
        """
        candidates: list[str] = []
        if not brief.sequence:
            return candidates

        candidates.append(brief.sequence)

        for effect in variant_effects:
            if effect.get("predicted_effect") == "stabilizing" and brief.sequence:
                mut = effect.get("mutation", "")
                if len(mut) >= 3:
                    try:
                        pos = int(mut[1:-1])
                        mut_aa = mut[-1]
                        if 1 <= pos <= len(brief.sequence):
                            mutated = brief.sequence[: pos - 1] + mut_aa + brief.sequence[pos:]
                            candidates.append(mutated)
                    except ValueError:
                        pass

        return candidates

    # ------------------------------------------------------------------
    # Step 5 — rank by perplexity
    # ------------------------------------------------------------------

    async def _rank_candidates(self, candidates: list[str]) -> list[dict[str, Any]]:
        """Rank candidate sequences by ESM-2 pseudo-perplexity.

        Args:
            candidates: Amino-acid sequences to rank.

        Returns:
            Sorted list of dicts with sequence, perplexity, length.
        """
        scored: list[dict[str, Any]] = []
        for seq in candidates[: self._max_candidates]:
            if not self.toolkit.validate_sequence(seq):
                continue
            try:
                ppl = await self.toolkit.compute_perplexity(seq)
                scored.append(
                    {
                        "sequence": seq,
                        "perplexity": round(ppl, 2),
                        "length": len(seq),
                    }
                )
            except Exception:
                logger.debug("Perplexity computation failed for sequence len=%d", len(seq))
                scored.append(
                    {
                        "sequence": seq,
                        "perplexity": float("inf"),
                        "length": len(seq),
                    }
                )

        return sorted(scored, key=lambda x: x["perplexity"])

    # ------------------------------------------------------------------
    # Step 6 — report generation
    # ------------------------------------------------------------------

    async def _generate_report(
        self,
        query: str,
        brief: _DesignBrief,
        strategy_info: dict[str, Any],
        ranked: list[dict[str, Any]],
        variant_effects: list[dict[str, Any]],
        mpnn_designs: list[dict[str, Any]],
    ) -> str:
        """Generate a narrative Markdown report.

        Args:
            query: Original user query.
            brief: Parsed design brief.
            strategy_info: Strategy decision with rationale.
            ranked: Perplexity-ranked candidates.
            variant_effects: Variant effect predictions.
            mpnn_designs: ProteinMPNN design results.

        Returns:
            Markdown report string.
        """
        parts: list[str] = [
            f"Design request: {query}\n",
            f"Strategy: {brief.strategy}\n",
            f"Rationale: {strategy_info.get('rationale', 'N/A')}\n",
        ]

        if ranked:
            parts.append(f"\nTop {min(5, len(ranked))} candidates by perplexity:\n")
            for i, cand in enumerate(ranked[:5], 1):
                seq_preview = (
                    cand["sequence"][:30] + "…" if len(cand["sequence"]) > 30 else cand["sequence"]
                )
                parts.append(
                    f"  {i}. {seq_preview} (len={cand['length']}, ppl={cand['perplexity']})\n"
                )

        if variant_effects:
            parts.append("\nVariant effects:\n")
            for ve in variant_effects[:10]:
                parts.append(
                    f"  - {ve['mutation']}: {ve['predicted_effect']} "
                    f"(Δll={ve['delta_log_likelihood']:.3f})\n"
                )

        if mpnn_designs:
            parts.append("\nProteinMPNN designs:\n")
            for d in mpnn_designs[:5]:
                parts.append(
                    f"  - score={d.get('score', 0):.2f}, recovery={d.get('recovery', 0):.1%}\n"
                )

        prompt = "".join(parts)

        return await self.ollama.generate(
            model=self.config.model_reasoner,
            prompt=prompt,
            system=PROMPTS["architect_report"],
        )

    # ------------------------------------------------------------------
    # Step 7 — KG updates
    # ------------------------------------------------------------------

    @staticmethod
    def _build_kg_updates(
        brief: _DesignBrief, ranked: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build KG triple dicts for designed proteins.

        Args:
            brief: Design brief.
            ranked: Ranked candidate sequences.

        Returns:
            List of triple dicts for :meth:`BaseAgent.update_kg`.
        """
        updates: list[dict[str, Any]] = []
        for i, cand in enumerate(ranked[:3]):
            updates.append(
                {
                    "source": {
                        "type": "DesignedProtein",
                        "properties": {
                            "name": f"designed_{brief.strategy}_{i + 1}",
                            "sequence": cand["sequence"],
                            "perplexity": cand["perplexity"],
                            "length": cand["length"],
                        },
                    },
                    "relation": "DESIGNED_FOR",
                    "target": {
                        "type": "Function",
                        "properties": {"name": brief.target_function},
                    },
                }
            )
        return updates

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(ranked: list[dict[str, Any]]) -> float:
        """Estimate confidence from candidate quality.

        Args:
            ranked: Perplexity-ranked candidates.

        Returns:
            Confidence score 0.0–0.95.
        """
        if not ranked:
            return 0.1

        best_ppl = ranked[0].get("perplexity", float("inf"))
        if best_ppl == float("inf"):
            return 0.1

        # Lower perplexity → higher confidence.  Typical protein ppl: 3–15.
        if best_ppl < 5:
            return 0.85
        if best_ppl < 10:
            return 0.7
        if best_ppl < 20:
            return 0.5
        return 0.3
