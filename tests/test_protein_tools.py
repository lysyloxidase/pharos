"""Tests for pharos.tools.protein_tools — ProteinToolkit with mocked models."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from unittest.mock import patch

import pytest

from pharos.tools.protein_tools import (
    DesignedSequence,
    ProteinToolkit,
    StructurePrediction,
    VariantEffect,
)

# ------------------------------------------------------------------
# Model tests
# ------------------------------------------------------------------


class TestVariantEffectModel:
    def test_defaults(self) -> None:
        v = VariantEffect(mutation="A23G")
        assert v.delta_log_likelihood == 0.0
        assert v.predicted_effect == "neutral"

    def test_custom(self) -> None:
        v = VariantEffect(
            mutation="L45P", delta_log_likelihood=-1.5, predicted_effect="stabilizing"
        )
        assert v.mutation == "L45P"
        assert v.predicted_effect == "stabilizing"


class TestDesignedSequenceModel:
    def test_defaults(self) -> None:
        d = DesignedSequence(sequence="ACDEF")
        assert d.score == 0.0
        assert d.recovery == 0.0


class TestStructurePredictionModel:
    def test_defaults(self) -> None:
        s = StructurePrediction(pdb_path="/tmp/test.pdb")
        assert s.plddt_mean == 0.0
        assert s.sequence_length == 0


# ------------------------------------------------------------------
# Static utility tests
# ------------------------------------------------------------------


class TestValidateSequence:
    def test_valid(self) -> None:
        assert ProteinToolkit.validate_sequence("ACDEFGHIKLMNPQRSTVWY") is True

    def test_lowercase_valid(self) -> None:
        assert ProteinToolkit.validate_sequence("acdef") is True

    def test_invalid_char(self) -> None:
        assert ProteinToolkit.validate_sequence("ACDEFX") is False

    def test_empty(self) -> None:
        assert ProteinToolkit.validate_sequence("") is False

    def test_numbers(self) -> None:
        assert ProteinToolkit.validate_sequence("ACD123") is False


class TestSequenceIdentity:
    def test_identical(self) -> None:
        assert ProteinToolkit.compute_sequence_identity("ACDEF", "ACDEF") == pytest.approx(1.0)

    def test_no_match(self) -> None:
        assert ProteinToolkit.compute_sequence_identity("AAAA", "CCCC") == pytest.approx(0.0)

    def test_partial(self) -> None:
        assert ProteinToolkit.compute_sequence_identity("ACDE", "ACFF") == pytest.approx(0.5)

    def test_different_lengths(self) -> None:
        identity = ProteinToolkit.compute_sequence_identity("ACDEF", "ACD")
        assert identity == pytest.approx(1.0)  # 3/3 match in shorter

    def test_empty(self) -> None:
        assert ProteinToolkit.compute_sequence_identity("", "ACDEF") == pytest.approx(0.0)


# ------------------------------------------------------------------
# ESM-2 embedding (mocked)
# ------------------------------------------------------------------


class TestComputeEmbedding:
    async def test_returns_embedding(self) -> None:
        import numpy as np

        toolkit = ProteinToolkit()
        fake_emb = np.array([0.1, 0.2, 0.3])
        with patch.object(toolkit, "_esm_embed_sync", return_value=fake_emb):
            result = await toolkit.compute_embedding("ACDEF")
        assert result is fake_emb

    async def test_calls_sync_impl(self) -> None:
        import numpy as np

        toolkit = ProteinToolkit()
        fake_emb = np.array([1.0, 2.0])
        with patch.object(toolkit, "_esm_embed_sync", return_value=fake_emb) as mock_sync:
            await toolkit.compute_embedding("MKTL")
        mock_sync.assert_called_once_with("MKTL")


# ------------------------------------------------------------------
# Variant effect prediction (mocked)
# ------------------------------------------------------------------


class TestPredictVariantEffects:
    async def test_returns_effects(self) -> None:
        toolkit = ProteinToolkit()
        expected = [
            VariantEffect(mutation="A1G", delta_log_likelihood=0.3, predicted_effect="neutral")
        ]
        with patch.object(toolkit, "_variant_effects_sync", return_value=expected):
            result = await toolkit.predict_variant_effects("ACDEF", ["A1G"])
        assert len(result) == 1
        assert result[0].mutation == "A1G"

    async def test_multiple_mutations(self) -> None:
        toolkit = ProteinToolkit()
        expected = [
            VariantEffect(mutation="A1G", predicted_effect="neutral"),
            VariantEffect(mutation="C2W", predicted_effect="destabilizing"),
        ]
        with patch.object(toolkit, "_variant_effects_sync", return_value=expected):
            result = await toolkit.predict_variant_effects("ACDEF", ["A1G", "C2W"])
        assert len(result) == 2


# ------------------------------------------------------------------
# ProtGPT2 generation (mocked)
# ------------------------------------------------------------------


class TestGenerateSequences:
    async def test_returns_sequences(self) -> None:
        toolkit = ProteinToolkit()
        fake_seqs = ["ACDEFGHIKLM", "MNPQRSTVWY"]
        with patch.object(toolkit, "_generate_protgpt2_sync", return_value=fake_seqs):
            result = await toolkit.generate_sequences(n=2)
        assert len(result) == 2
        assert all(ProteinToolkit.validate_sequence(s) for s in result)

    async def test_passes_params(self) -> None:
        toolkit = ProteinToolkit()
        with patch.object(toolkit, "_generate_protgpt2_sync", return_value=[]) as mock_gen:
            await toolkit.generate_sequences(n=5, max_length=100, temperature=0.8)
        mock_gen.assert_called_once_with(5, 100, 0.8)


# ------------------------------------------------------------------
# ProteinMPNN (mocked)
# ------------------------------------------------------------------


class TestDesignSequenceForStructure:
    async def test_raises_without_mpnn(self) -> None:
        toolkit = ProteinToolkit(proteinmpnn_path=None)
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="ProteinMPNN"),
        ):
            await toolkit.design_sequence_for_structure("/fake/test.pdb")

    async def test_with_mpnn_path(self, tmp_path: Path) -> None:
        mpnn_dir = tmp_path / "ProteinMPNN"
        mpnn_dir.mkdir()
        toolkit = ProteinToolkit(proteinmpnn_path=str(mpnn_dir))

        expected = [DesignedSequence(sequence="ACDEF", score=-1.2, recovery=0.45)]
        with patch.object(toolkit, "_run_proteinmpnn_sync", return_value=expected):
            result = await toolkit.design_sequence_for_structure("/fake/test.pdb", n_designs=4)
        assert len(result) == 1
        assert result[0].sequence == "ACDEF"


class TestParseMpnnOutput:
    def test_parses_fasta(self, tmp_path: Path) -> None:
        seqs_dir = tmp_path / "seqs"
        seqs_dir.mkdir()
        fasta = seqs_dir / "test.fa"
        fasta.write_text(
            ">design_1, score=1.23, seq_recovery=0.65\n"
            "ACDEFGHIKLM\n"
            ">design_2, score=0.99, seq_recovery=0.78\n"
            "MNPQRSTVWY\n"
        )
        designs = ProteinToolkit._parse_mpnn_output(tmp_path)
        assert len(designs) == 2
        assert designs[0].sequence == "ACDEFGHIKLM"
        assert designs[0].score == pytest.approx(1.23)
        assert designs[0].recovery == pytest.approx(0.65)
        assert designs[1].sequence == "MNPQRSTVWY"

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert ProteinToolkit._parse_mpnn_output(tmp_path) == []


# ------------------------------------------------------------------
# ESMFold (mocked)
# ------------------------------------------------------------------


class TestPredictStructure:
    async def test_returns_prediction(self) -> None:
        toolkit = ProteinToolkit()
        expected = StructurePrediction(
            pdb_path="/tmp/out.pdb", plddt_mean=75.0, sequence_length=100, time_seconds=120.0
        )
        with patch.object(toolkit, "_predict_structure_sync", return_value=expected):
            result = await toolkit.predict_structure("ACDEF" * 20, "/tmp/out.pdb")
        assert result.pdb_path == "/tmp/out.pdb"
        assert result.plddt_mean == 75.0


# ------------------------------------------------------------------
# Perplexity (mocked)
# ------------------------------------------------------------------


class TestComputePerplexity:
    async def test_returns_float(self) -> None:
        toolkit = ProteinToolkit()
        with patch.object(toolkit, "_perplexity_sync", return_value=5.5):
            result = await toolkit.compute_perplexity("ACDEF")
        assert result == pytest.approx(5.5)

    async def test_lower_is_better(self) -> None:
        toolkit = ProteinToolkit()
        with patch.object(toolkit, "_perplexity_sync", side_effect=[3.0, 12.0]):
            good = await toolkit.compute_perplexity("NATURAL")
            bad = await toolkit.compute_perplexity("XZJBQO")
        assert good < bad
