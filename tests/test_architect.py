"""Tests for pharos.agents.architect — protein design pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.agents.architect import ArchitectAgent, _DesignBrief
from pharos.config import Settings
from pharos.orchestration.task_models import Task, TaskType, WorkflowState
from pharos.tools.protein_tools import (
    DesignedSequence,
    ProteinToolkit,
    StructurePrediction,
    VariantEffect,
)

# ------------------------------------------------------------------
# Sample LLM responses
# ------------------------------------------------------------------

SAMPLE_BRIEF_JSON = json.dumps(
    {
        "strategy": "de_novo",
        "target_function": "thermostable lipase",
        "constraints": ["thermostable", "pH 9"],
        "sequence": None,
        "pdb_path": None,
        "mutations_of_interest": [],
    }
)

SAMPLE_BRIEF_REDESIGN_JSON = json.dumps(
    {
        "strategy": "redesign",
        "target_function": "lipase",
        "constraints": [],
        "sequence": None,
        "pdb_path": "/data/lipase.pdb",
        "mutations_of_interest": [],
    }
)

SAMPLE_BRIEF_MUTAGENESIS_JSON = json.dumps(
    {
        "strategy": "mutagenesis",
        "target_function": "improve stability",
        "constraints": ["thermostable"],
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "pdb_path": None,
        "mutations_of_interest": ["A1G", "D4W"],
    }
)

SAMPLE_STRATEGY_JSON = json.dumps(
    {
        "strategy": "de_novo",
        "rationale": "No starting sequence provided, de novo is appropriate.",
        "target_properties": ["thermostability", "pH tolerance"],
        "suggested_mutations": [],
    }
)

SAMPLE_STRATEGY_MUTAGENESIS_JSON = json.dumps(
    {
        "strategy": "mutagenesis",
        "rationale": "Sequence provided; directed mutagenesis is most efficient.",
        "target_properties": ["stability"],
        "suggested_mutations": ["A1G", "D4W"],
    }
)

SAMPLE_REPORT = (
    "# Protein Design Report\n\n"
    "## Design Brief\n"
    "Design a thermostable lipase for pH 9.\n\n"
    "## Strategy\n"
    "De novo generation via ProtGPT2.\n\n"
    "## Candidates\n"
    "1. ACDEFGHIKLM (ppl=4.2)\n"
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.architect_max_candidates = 5
    s.proteinmpnn_path = ""
    return s


@pytest.fixture
def mock_toolkit() -> MagicMock:
    tk = MagicMock(spec=ProteinToolkit)
    tk.generate_sequences = AsyncMock(return_value=["ACDEFGHIKLM", "MNPQRSTVWY"])
    tk.compute_perplexity = AsyncMock(return_value=5.0)
    tk.compute_embedding = AsyncMock(return_value=[0.1] * 10)
    tk.predict_variant_effects = AsyncMock(return_value=[])
    tk.predict_structure = AsyncMock(
        return_value=StructurePrediction(
            pdb_path="/tmp/out.pdb", plddt_mean=80.0, sequence_length=50, time_seconds=60.0
        )
    )
    tk.design_sequence_for_structure = AsyncMock(
        return_value=[DesignedSequence(sequence="ACDEF", score=-1.0, recovery=0.5)]
    )
    tk.validate_sequence = MagicMock(side_effect=ProteinToolkit.validate_sequence)
    return tk


@pytest.fixture
def architect(
    mock_ollama: MagicMock,
    mock_neo4j: MagicMock,
    settings: Settings,
    mock_toolkit: MagicMock,
) -> ArchitectAgent:
    return ArchitectAgent(
        ollama=mock_ollama,
        kg=mock_neo4j,
        config=settings,
        toolkit=mock_toolkit,
    )


@pytest.fixture
def state() -> WorkflowState:
    return WorkflowState(
        task=Task(query="test", task_type=TaskType.DESIGN_PROTEIN),
        results=[],
        current_agent="architect",
        kg_context="",
        iteration=0,
    )


# ------------------------------------------------------------------
# Design brief parsing
# ------------------------------------------------------------------


class TestArchitectParseBrief:
    async def test_parses_de_novo(self, architect: ArchitectAgent) -> None:
        architect.ollama.generate = AsyncMock(return_value=SAMPLE_BRIEF_JSON)
        brief = await architect._parse_brief("design a thermostable lipase for pH 9")
        assert brief.strategy == "de_novo"
        assert brief.target_function == "thermostable lipase"
        assert "thermostable" in brief.constraints

    async def test_parses_mutagenesis(self, architect: ArchitectAgent) -> None:
        architect.ollama.generate = AsyncMock(return_value=SAMPLE_BRIEF_MUTAGENESIS_JSON)
        brief = await architect._parse_brief("mutate A1G and D4W")
        assert brief.strategy == "mutagenesis"
        assert brief.sequence == "ACDEFGHIKLMNPQRSTVWY"
        assert "A1G" in brief.mutations_of_interest

    async def test_fallback_on_failure(self, architect: ArchitectAgent) -> None:
        architect.ollama.generate = AsyncMock(side_effect=Exception("LLM down"))
        brief = await architect._parse_brief("some query")
        assert brief.strategy == "de_novo"
        assert brief.target_function == "some query"


# ------------------------------------------------------------------
# Strategy decision
# ------------------------------------------------------------------


class TestArchitectStrategy:
    async def test_decides_strategy(self, architect: ArchitectAgent) -> None:
        architect.ollama.generate = AsyncMock(return_value=SAMPLE_STRATEGY_JSON)
        brief = _DesignBrief(target_function="lipase")
        result = await architect._decide_strategy(brief, "KG context here")
        assert result["strategy"] == "de_novo"
        assert "rationale" in result

    async def test_fallback_on_failure(self, architect: ArchitectAgent) -> None:
        architect.ollama.generate = AsyncMock(side_effect=Exception("fail"))
        brief = _DesignBrief(strategy="mutagenesis")
        result = await architect._decide_strategy(brief, "")
        assert result["strategy"] == "mutagenesis"


# ------------------------------------------------------------------
# De novo pipeline
# ------------------------------------------------------------------


class TestArchitectDeNovo:
    async def test_generates_candidates(
        self, architect: ArchitectAgent, mock_toolkit: MagicMock
    ) -> None:
        brief = _DesignBrief(strategy="de_novo")
        candidates = await architect._run_de_novo(brief)
        assert len(candidates) == 2
        mock_toolkit.generate_sequences.assert_called_once()

    async def test_handles_failure(
        self, architect: ArchitectAgent, mock_toolkit: MagicMock
    ) -> None:
        mock_toolkit.generate_sequences = AsyncMock(side_effect=Exception("model not loaded"))
        brief = _DesignBrief(strategy="de_novo")
        candidates = await architect._run_de_novo(brief)
        assert candidates == []


# ------------------------------------------------------------------
# Redesign pipeline
# ------------------------------------------------------------------


class TestArchitectRedesign:
    async def test_runs_mpnn(self, architect: ArchitectAgent, mock_toolkit: MagicMock) -> None:
        brief = _DesignBrief(strategy="redesign", pdb_path="/data/lipase.pdb")
        designs = await architect._run_redesign(brief)
        assert len(designs) == 1
        assert designs[0]["sequence"] == "ACDEF"
        mock_toolkit.design_sequence_for_structure.assert_called_once()

    async def test_no_pdb_returns_empty(self, architect: ArchitectAgent) -> None:
        brief = _DesignBrief(strategy="redesign", pdb_path=None)
        designs = await architect._run_redesign(brief)
        assert designs == []

    async def test_handles_mpnn_not_installed(
        self, architect: ArchitectAgent, mock_toolkit: MagicMock
    ) -> None:
        mock_toolkit.design_sequence_for_structure = AsyncMock(
            side_effect=RuntimeError("ProteinMPNN not installed")
        )
        brief = _DesignBrief(strategy="redesign", pdb_path="/data/test.pdb")
        designs = await architect._run_redesign(brief)
        assert designs == []


# ------------------------------------------------------------------
# Mutagenesis pipeline
# ------------------------------------------------------------------


class TestArchitectMutagenesis:
    async def test_scores_mutations(
        self, architect: ArchitectAgent, mock_toolkit: MagicMock
    ) -> None:
        mock_toolkit.predict_variant_effects = AsyncMock(
            return_value=[
                VariantEffect(
                    mutation="A1G", delta_log_likelihood=-0.8, predicted_effect="stabilizing"
                ),
                VariantEffect(
                    mutation="D4W", delta_log_likelihood=1.2, predicted_effect="destabilizing"
                ),
            ]
        )
        brief = _DesignBrief(
            strategy="mutagenesis",
            sequence="ACDEFGHIKLMNPQRSTVWY",
            mutations_of_interest=["A1G", "D4W"],
        )
        effects = await architect._run_mutagenesis(brief)
        assert len(effects) == 2
        assert effects[0]["mutation"] == "A1G"

    async def test_no_sequence_returns_empty(self, architect: ArchitectAgent) -> None:
        brief = _DesignBrief(strategy="mutagenesis", sequence=None, mutations_of_interest=["A1G"])
        effects = await architect._run_mutagenesis(brief)
        assert effects == []

    async def test_no_mutations_returns_empty(self, architect: ArchitectAgent) -> None:
        brief = _DesignBrief(strategy="mutagenesis", sequence="ACDEF", mutations_of_interest=[])
        effects = await architect._run_mutagenesis(brief)
        assert effects == []


class TestCandidatesFromMutagenesis:
    def test_applies_stabilizing_mutations(self) -> None:
        brief = _DesignBrief(sequence="ACDEF")
        effects = [
            {"mutation": "A1G", "predicted_effect": "stabilizing", "delta_log_likelihood": -1.0},
            {"mutation": "C2W", "predicted_effect": "destabilizing", "delta_log_likelihood": 1.0},
        ]
        candidates = ArchitectAgent._candidates_from_mutagenesis(brief, effects)
        assert candidates[0] == "ACDEF"  # WT
        assert len(candidates) == 2  # WT + 1 stabilizing
        assert candidates[1] == "GCDEF"  # A1G applied

    def test_no_sequence_returns_empty(self) -> None:
        brief = _DesignBrief(sequence=None)
        assert ArchitectAgent._candidates_from_mutagenesis(brief, []) == []


# ------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------


class TestArchitectRanking:
    async def test_ranks_by_perplexity(
        self, architect: ArchitectAgent, mock_toolkit: MagicMock
    ) -> None:
        mock_toolkit.compute_perplexity = AsyncMock(side_effect=[10.0, 3.0, 7.0])
        ranked = await architect._rank_candidates(["SEQAAA", "SEQBBB", "SEQCCC"])
        assert ranked[0]["perplexity"] == 3.0
        assert ranked[-1]["perplexity"] == 10.0

    async def test_empty_candidates(self, architect: ArchitectAgent) -> None:
        ranked = await architect._rank_candidates([])
        assert ranked == []


# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------


class TestArchitectRun:
    async def test_full_de_novo_pipeline(
        self,
        architect: ArchitectAgent,
        state: WorkflowState,
        mock_toolkit: MagicMock,
    ) -> None:
        architect.ollama.generate = AsyncMock(
            side_effect=[
                SAMPLE_BRIEF_JSON,  # parse brief
                SAMPLE_STRATEGY_JSON,  # strategy decision
                SAMPLE_REPORT,  # report generation
            ]
        )

        task = Task(query="design a thermostable lipase for pH 9")
        result = await architect.run(task, state)

        assert result.agent_name == "Architect"
        assert result.content == SAMPLE_REPORT
        assert result.artifacts["strategy"] == "de_novo"
        assert len(result.artifacts["candidates"]) > 0
        assert result.confidence > 0

    async def test_mutagenesis_pipeline(
        self,
        architect: ArchitectAgent,
        state: WorkflowState,
        mock_toolkit: MagicMock,
    ) -> None:
        mock_toolkit.predict_variant_effects = AsyncMock(
            return_value=[
                VariantEffect(
                    mutation="A1G", delta_log_likelihood=-0.8, predicted_effect="stabilizing"
                ),
            ]
        )

        architect.ollama.generate = AsyncMock(
            side_effect=[
                SAMPLE_BRIEF_MUTAGENESIS_JSON,
                SAMPLE_STRATEGY_MUTAGENESIS_JSON,
                "Mutagenesis report.",
            ]
        )

        task = Task(query="mutate A1G in my sequence")
        result = await architect.run(task, state)

        assert result.agent_name == "Architect"
        assert result.artifacts["strategy"] == "mutagenesis"
        assert len(result.artifacts["variant_effects"]) > 0

    async def test_confidence_high_on_low_perplexity(self, architect: ArchitectAgent) -> None:
        ranked = [{"sequence": "ACDEF", "perplexity": 3.5, "length": 5}]
        confidence = architect._compute_confidence(ranked)
        assert confidence == 0.85

    async def test_confidence_low_on_empty(self, architect: ArchitectAgent) -> None:
        assert architect._compute_confidence([]) == 0.1

    async def test_kg_updates_for_top_candidates(self, architect: ArchitectAgent) -> None:
        brief = _DesignBrief(strategy="de_novo", target_function="lipase")
        ranked = [
            {"sequence": "ACDEF", "perplexity": 4.0, "length": 5},
            {"sequence": "GHIKL", "perplexity": 6.0, "length": 5},
        ]
        updates = architect._build_kg_updates(brief, ranked)
        assert len(updates) == 2
        assert updates[0]["relation"] == "DESIGNED_FOR"
        assert updates[0]["source"]["type"] == "DesignedProtein"
        assert updates[0]["target"]["properties"]["name"] == "lipase"


# ------------------------------------------------------------------
# KG context query
# ------------------------------------------------------------------


class TestArchitectKgContext:
    async def test_returns_context(self, architect: ArchitectAgent, mock_neo4j: MagicMock) -> None:
        mock_neo4j.query = AsyncMock(
            side_effect=[
                [{"name": "LipaseA", "function": "lipase activity"}],
                [{"mutation": "G45A", "protein": "LipaseA"}],
            ]
        )
        brief = _DesignBrief(target_function="lipase")
        ctx = await architect._query_kg_context(brief)
        assert "LipaseA" in ctx
        assert "G45A" in ctx

    async def test_handles_kg_failure(
        self, architect: ArchitectAgent, mock_neo4j: MagicMock
    ) -> None:
        mock_neo4j.query = AsyncMock(side_effect=Exception("KG down"))
        brief = _DesignBrief(target_function="lipase")
        ctx = await architect._query_kg_context(brief)
        assert "No relevant KG context" in ctx
