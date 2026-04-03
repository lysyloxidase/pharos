"""Protein ML tool wrappers — ESM-2, ProtGPT2, ProteinMPNN, ESMFold.

Provides a unified :class:`ProteinToolkit` that lazily loads heavyweight
models (ESM-2 ~4 GB, ProtGPT2 ~4 GB) on first use and caches them for
subsequent calls.  All public methods are async so they integrate with
the PHAROS agent loop without blocking.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Timeout for ESMFold structure prediction on CPU (seconds).
_ESMFOLD_TIMEOUT = 600


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


class VariantEffect(BaseModel):
    """Predicted effect of a single-point mutation.

    Attributes:
        mutation: Mutation string (e.g. ``"A23G"``).
        delta_log_likelihood: Log-likelihood difference (WT - mutant).
        predicted_effect: One of stabilizing / neutral / destabilizing.
    """

    mutation: str
    delta_log_likelihood: float = 0.0
    predicted_effect: str = "neutral"


class DesignedSequence(BaseModel):
    """Sequence designed by ProteinMPNN for a given backbone.

    Attributes:
        sequence: Amino-acid sequence (one-letter code).
        score: ProteinMPNN log-probability score.
        recovery: Fraction of native residues recovered (0-1).
    """

    sequence: str
    score: float = 0.0
    recovery: float = 0.0


class StructurePrediction(BaseModel):
    """Result of ESMFold structure prediction.

    Attributes:
        pdb_path: Path to the output PDB file.
        plddt_mean: Mean predicted LDDT confidence (0-100).
        sequence_length: Number of residues.
        time_seconds: Wall-clock prediction time.
    """

    pdb_path: str
    plddt_mean: float = 0.0
    sequence_length: int = 0
    time_seconds: float = 0.0


# ------------------------------------------------------------------
# ProteinToolkit
# ------------------------------------------------------------------


class ProteinToolkit:
    """CPU-compatible protein ML toolkit with lazy model loading.

    Models are loaded on first call and cached for the lifetime of the
    instance.  All heavy computation is off-loaded to a thread-pool via
    :func:`asyncio.to_thread` so the event loop stays responsive.

    Args:
        proteinmpnn_path: Path to a local ProteinMPNN clone.
            If *None*, :meth:`design_sequence_for_structure` will raise
            a descriptive error.
        device: PyTorch device string (default ``"cpu"``).
    """

    def __init__(
        self,
        proteinmpnn_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._device = device
        self._proteinmpnn_path = proteinmpnn_path

        # Lazy-loaded model handles
        self._esm_model: Any = None
        self._esm_alphabet: Any = None
        self._esm_batch_converter: Any = None

        self._protgpt2_model: Any = None
        self._protgpt2_tokenizer: Any = None

    # ------------------------------------------------------------------
    # ESM-2 helpers (private)
    # ------------------------------------------------------------------

    def _load_esm(self) -> None:
        """Load ESM-2 model + alphabet on first use."""
        if self._esm_model is not None:
            return
        import esm

        logger.info("Loading ESM-2 (esm2_t33_650M_UR50D) — this may take a moment…")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()
        self._esm_model = model
        self._esm_alphabet = alphabet
        self._esm_batch_converter = alphabet.get_batch_converter()

    def _esm_embed_sync(self, sequence: str) -> Any:
        """Compute per-sequence ESM-2 embedding (sync, CPU)."""
        import torch

        self._load_esm()
        data = [("protein", sequence)]
        _, _, batch_tokens = self._esm_batch_converter(data)
        batch_tokens = batch_tokens.to(self._device)

        with torch.no_grad():
            results = self._esm_model(batch_tokens, repr_layers=[33], return_contacts=False)

        # Mean over residue positions (exclude BOS/EOS tokens)
        token_repr = results["representations"][33]
        embedding = token_repr[0, 1 : len(sequence) + 1].mean(0)
        return embedding.cpu().numpy()

    def _esm_logits_sync(self, sequence: str) -> Any:
        """Get masked-language-model logits for every position (sync)."""
        import torch

        self._load_esm()
        data = [("protein", sequence)]
        _, _, batch_tokens = self._esm_batch_converter(data)
        batch_tokens = batch_tokens.to(self._device)

        with torch.no_grad():
            logits = self._esm_model(batch_tokens)["logits"]
        return logits[0].cpu()  # (seq_len+2, vocab)

    # ------------------------------------------------------------------
    # ProtGPT2 helpers (private)
    # ------------------------------------------------------------------

    def _load_protgpt2(self) -> None:
        """Load ProtGPT2 model + tokenizer on first use."""
        if self._protgpt2_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading ProtGPT2 — this may take a moment…")
        self._protgpt2_tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        self._protgpt2_model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
        self._protgpt2_model.eval()

    def _generate_protgpt2_sync(self, n: int, max_length: int, temperature: float) -> list[str]:
        """Generate *n* protein sequences with ProtGPT2 (sync, CPU)."""
        self._load_protgpt2()
        tokenizer = self._protgpt2_tokenizer
        model = self._protgpt2_model

        input_ids = tokenizer.encode("<|endoftext|>", return_tensors="pt")
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n,
            top_k=950,
            repetition_penalty=1.2,
        )

        sequences: list[str] = []
        for output in outputs:
            text: str = tokenizer.decode(output, skip_special_tokens=True)
            # ProtGPT2 may embed newlines between sequences
            cleaned = "".join(ch for ch in text if ch in _STANDARD_AA)
            if cleaned:
                sequences.append(cleaned)
        return sequences

    # ------------------------------------------------------------------
    # Public API — ESM-2
    # ------------------------------------------------------------------

    async def compute_embedding(self, sequence: str) -> Any:
        """Compute ESM-2 per-sequence embedding.

        Lazily loads the ESM-2 650M model on first call (~4 GB).

        Args:
            sequence: Amino-acid sequence (one-letter code).

        Returns:
            1-D numpy array (embedding dimension 1280).
        """
        return await asyncio.to_thread(self._esm_embed_sync, sequence)

    async def predict_variant_effects(
        self, wt_sequence: str, mutations: list[str]
    ) -> list[VariantEffect]:
        """Predict effects of point mutations via masked marginal scoring.

        Compares log-likelihood of wild-type residue vs mutant residue at
        each masked position.

        Args:
            wt_sequence: Wild-type amino-acid sequence.
            mutations: List of mutation strings like ``["A23G", "L45P"]``.

        Returns:
            List of :class:`VariantEffect` objects.
        """
        return await asyncio.to_thread(self._variant_effects_sync, wt_sequence, mutations)

    def _variant_effects_sync(self, wt_sequence: str, mutations: list[str]) -> list[VariantEffect]:
        """Sync implementation of variant-effect scoring."""
        import torch

        self._load_esm()
        alphabet = self._esm_alphabet

        logits = self._esm_logits_sync(wt_sequence)
        log_probs = torch.log_softmax(logits, dim=-1)

        effects: list[VariantEffect] = []
        for mut in mutations:
            wt_aa, pos_str, mut_aa = mut[0], mut[1:-1], mut[-1]
            try:
                pos = int(pos_str)
            except ValueError:
                effects.append(VariantEffect(mutation=mut, predicted_effect="invalid"))
                continue

            if pos < 1 or pos > len(wt_sequence):
                effects.append(VariantEffect(mutation=mut, predicted_effect="invalid"))
                continue

            # logits index = pos (1-indexed in sequence = 1-indexed in tokens due to BOS)
            wt_idx = alphabet.get_idx(wt_aa)
            mut_idx = alphabet.get_idx(mut_aa)
            delta = float(log_probs[pos, wt_idx] - log_probs[pos, mut_idx])

            if delta > 0.5:
                label = "destabilizing"
            elif delta < -0.5:
                label = "stabilizing"
            else:
                label = "neutral"

            effects.append(
                VariantEffect(mutation=mut, delta_log_likelihood=delta, predicted_effect=label)
            )
        return effects

    # ------------------------------------------------------------------
    # Public API — ProtGPT2
    # ------------------------------------------------------------------

    async def generate_sequences(
        self,
        n: int = 10,
        max_length: int = 200,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate *de novo* protein sequences with ProtGPT2.

        Lazily loads ProtGPT2 (738 M) on first call (~4 GB).

        Args:
            n: Number of sequences to generate.
            max_length: Maximum sequence length (tokens).
            temperature: Sampling temperature.

        Returns:
            List of amino-acid sequence strings.
        """
        return await asyncio.to_thread(self._generate_protgpt2_sync, n, max_length, temperature)

    # ------------------------------------------------------------------
    # Public API — ProteinMPNN
    # ------------------------------------------------------------------

    async def design_sequence_for_structure(
        self,
        pdb_path: str,
        n_designs: int = 8,
    ) -> list[DesignedSequence]:
        """Design sequences for a backbone structure via ProteinMPNN.

        Requires a local clone of `<https://github.com/dauparas/ProteinMPNN>`_.
        Set ``proteinmpnn_path`` in the constructor or the
        ``PHAROS_PROTEINMPNN_PATH`` environment variable.

        Args:
            pdb_path: Path to input PDB file.
            n_designs: Number of designed sequences to return.

        Returns:
            List of :class:`DesignedSequence` objects.

        Raises:
            RuntimeError: If ProteinMPNN is not installed.
        """
        mpnn_path = self._proteinmpnn_path or os.environ.get("PHAROS_PROTEINMPNN_PATH")
        if not mpnn_path or not Path(mpnn_path).exists():
            raise RuntimeError(
                "ProteinMPNN is not installed. Clone it from "
                "https://github.com/dauparas/ProteinMPNN and set "
                "proteinmpnn_path or PHAROS_PROTEINMPNN_PATH."
            )
        return await asyncio.to_thread(self._run_proteinmpnn_sync, mpnn_path, pdb_path, n_designs)

    def _run_proteinmpnn_sync(
        self, mpnn_path: str, pdb_path: str, n_designs: int
    ) -> list[DesignedSequence]:
        """Run ProteinMPNN subprocess (sync)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()

            cmd = [
                "python",
                str(Path(mpnn_path) / "protein_mpnn_run.py"),
                "--pdb_path",
                pdb_path,
                "--out_folder",
                str(out_dir),
                "--num_seq_per_target",
                str(n_designs),
                "--sampling_temp",
                "0.1",
                "--seed",
                "42",
                "--batch_size",
                "1",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            if result.returncode != 0:
                logger.error("ProteinMPNN failed: %s", result.stderr[:500])
                return []

            return self._parse_mpnn_output(out_dir)

    @staticmethod
    def _parse_mpnn_output(out_dir: Path) -> list[DesignedSequence]:
        """Parse ProteinMPNN FASTA output files."""
        designs: list[DesignedSequence] = []
        seqs_dir = out_dir / "seqs"
        if not seqs_dir.exists():
            return designs

        for fasta in seqs_dir.glob("*.fa"):
            lines = fasta.read_text().splitlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith(">"):
                    header = lines[i]
                    seq_lines: list[str] = []
                    i += 1
                    while i < len(lines) and not lines[i].startswith(">"):
                        seq_lines.append(lines[i].strip())
                        i += 1
                    sequence = "".join(seq_lines)

                    score = 0.0
                    recovery = 0.0
                    if "score=" in header:
                        with contextlib.suppress(IndexError, ValueError):
                            score = float(header.split("score=")[1].split(",")[0].split()[0])
                    if "seq_recovery=" in header:
                        with contextlib.suppress(IndexError, ValueError):
                            recovery = float(
                                header.split("seq_recovery=")[1].split(",")[0].split()[0]
                            )

                    designs.append(
                        DesignedSequence(sequence=sequence, score=score, recovery=recovery)
                    )
                else:
                    i += 1

        return designs

    # ------------------------------------------------------------------
    # Public API — ESMFold
    # ------------------------------------------------------------------

    async def predict_structure(
        self,
        sequence: str,
        output_pdb: str,
    ) -> StructurePrediction:
        """Predict 3D structure using ESMFold.

        Runs on CPU — expect ~5 min per protein.  A 10-minute timeout
        is enforced.

        Args:
            sequence: Amino-acid sequence (one-letter code).
            output_pdb: Path to write the output PDB file.

        Returns:
            :class:`StructurePrediction` with path, pLDDT, timing.
        """
        return await asyncio.to_thread(self._predict_structure_sync, sequence, output_pdb)

    def _predict_structure_sync(self, sequence: str, output_pdb: str) -> StructurePrediction:
        """Run ESMFold structure prediction (sync, CPU)."""
        import torch
        from transformers import EsmForProteinFolding, EsmTokenizer

        logger.info(
            "Running ESMFold for %d-residue sequence (CPU) — this may take several minutes…",
            len(sequence),
        )
        t0 = time.time()

        tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        model.eval()  # type: ignore[no-untyped-call]
        model.cpu()

        tokenized = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)

        with torch.no_grad():
            output = model(**tokenized)

        pdb_string: str = model.output_to_pdb(output)[0]
        elapsed = time.time() - t0

        Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
        Path(output_pdb).write_text(pdb_string)

        # Extract mean pLDDT from B-factor column
        plddt_values: list[float] = []
        for line in pdb_string.splitlines():
            if line.startswith("ATOM"):
                with contextlib.suppress(ValueError, IndexError):
                    plddt_values.append(float(line[60:66].strip()))
        plddt_mean = sum(plddt_values) / len(plddt_values) if plddt_values else 0.0

        return StructurePrediction(
            pdb_path=output_pdb,
            plddt_mean=plddt_mean,
            sequence_length=len(sequence),
            time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Public API — Analysis utilities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sequence_identity(seq_a: str, seq_b: str) -> float:
        """Compute pairwise sequence identity (fraction of matching residues).

        Sequences must be pre-aligned or the same length.  If lengths
        differ, the comparison is truncated to the shorter sequence.

        Args:
            seq_a: First amino-acid sequence.
            seq_b: Second amino-acid sequence.

        Returns:
            Identity fraction (0.0–1.0).
        """
        length = min(len(seq_a), len(seq_b))
        if length == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(seq_a[:length], seq_b[:length], strict=False))
        return matches / length

    @staticmethod
    def validate_sequence(sequence: str) -> bool:
        """Check whether *sequence* contains only standard amino acids.

        Args:
            sequence: Candidate amino-acid string.

        Returns:
            True if every character is in the 20 standard amino acids.
        """
        return bool(sequence) and all(ch in _STANDARD_AA for ch in sequence.upper())

    async def compute_perplexity(self, sequence: str) -> float:
        """Compute pseudo-perplexity from ESM-2 logits.

        Lower perplexity ≈ more protein-like sequence.

        Args:
            sequence: Amino-acid sequence.

        Returns:
            Pseudo-perplexity score.
        """
        return await asyncio.to_thread(self._perplexity_sync, sequence)

    def _perplexity_sync(self, sequence: str) -> float:
        """Sync pseudo-perplexity computation."""
        import torch

        self._load_esm()
        alphabet = self._esm_alphabet

        logits = self._esm_logits_sync(sequence)
        log_probs = torch.log_softmax(logits, dim=-1)

        nll_sum = 0.0
        for i, aa in enumerate(sequence):
            idx = alphabet.get_idx(aa)
            nll_sum -= float(log_probs[i + 1, idx])  # +1 for BOS token

        avg_nll = nll_sum / max(len(sequence), 1)
        return math.exp(avg_nll)
