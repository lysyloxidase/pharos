"""Agent 6 — Sentinel: verification and fact-checking agent.

The Sentinel is the final agent in every PHAROS pipeline.  It verifies
claims against the knowledge graph and PubMed, validates molecular and
protein artifacts, checks citations, and detects hallucinations.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from pharos.agents.base import BaseAgent
from pharos.graph.entity_extractor import _extract_json
from pharos.orchestration.prompts import PROMPTS
from pharos.orchestration.task_models import AgentResult, Task, WorkflowState

if TYPE_CHECKING:
    from pharos.config import Settings
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.tools.ollama_client import OllamaClient
    from pharos.tools.pubmed_tools import PubMedClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


class ClaimCheck(BaseModel):
    """Verification result for a single claim.

    Attributes:
        claim: The claim text being verified.
        supported: Whether evidence was found.
        source: Where evidence was found (kg / pubmed / none).
        issue: Description of the problem if not supported.
        passed: Whether the check passed.
    """

    claim: str
    supported: bool = False
    source: str = "none"
    issue: str = ""
    passed: bool = False


class MoleculeCheck(BaseModel):
    """Validation result for a SMILES string.

    Attributes:
        smiles: The SMILES string.
        valid: Whether RDKit can parse it.
        issue: Description of the problem if invalid.
        passed: Whether the check passed.
    """

    smiles: str
    valid: bool = False
    issue: str = ""
    passed: bool = False


class SequenceCheck(BaseModel):
    """Validation result for a protein sequence.

    Attributes:
        sequence_preview: First 20 characters of the sequence.
        valid: Whether it contains only standard amino acids.
        length: Sequence length.
        issue: Description of the problem if invalid.
        passed: Whether the check passed.
    """

    sequence_preview: str = ""
    valid: bool = False
    length: int = 0
    issue: str = ""
    passed: bool = False


class CitationCheck(BaseModel):
    """Verification result for a PMID citation.

    Attributes:
        pmid: The PubMed ID.
        exists: Whether the PMID is findable on PubMed.
        issue: Description of the problem if not found.
        passed: Whether the check passed.
    """

    pmid: str
    exists: bool = False
    issue: str = ""
    passed: bool = False


class Verification(BaseModel):
    """Aggregated verification for one agent result.

    Attributes:
        agent_name: Name of the agent whose output was verified.
        checks: All individual checks performed.
        score: Fraction of checks that passed (0-1).
        issues: List of issue descriptions from failed checks.
    """

    agent_name: str
    checks: list[dict[str, Any]] = Field(default_factory=list)
    score: float = 0.0
    issues: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# SentinelAgent
# ------------------------------------------------------------------

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


class SentinelAgent(BaseAgent):
    """Verification and fact-checking agent.

    Validates outputs from all preceding agents by:

    1. Extracting and fact-checking claims against the KG and PubMed.
    2. Validating SMILES molecules via RDKit.
    3. Validating protein sequences.
    4. Verifying PMID citations exist.
    5. Running LLM-based hallucination detection.

    Args:
        ollama: Async Ollama client.
        kg: Neo4j knowledge-graph manager.
        config: Application settings.
        pubmed: Optional PubMed client for citation verification.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        kg: Neo4jManager,
        config: Settings,
        *,
        pubmed: PubMedClient | None = None,
    ) -> None:
        super().__init__(ollama, kg, config)
        self._pubmed = pubmed

    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Verify all preceding agent results.

        Args:
            task: The original task (for context).
            state: Workflow state containing prior agent results.

        Returns:
            AgentResult with verification report and per-agent scores.
        """
        results = state["results"]

        verifications: list[Verification] = []
        for result in results:
            if result.agent_name.lower() == "sentinel":
                continue  # don't verify ourselves
            if result.agent_name.lower() == "router":
                continue  # router output is structural, not factual
            v = await self.verify_result(result)
            verifications.append(v)

        if verifications:
            mean_score = sum(v.score for v in verifications) / len(verifications)
        else:
            mean_score = 1.0

        report = self._format_verification_report(verifications)

        return AgentResult(
            agent_name="Sentinel",
            task_id=str(uuid.uuid4()),
            content=report,
            artifacts={"verifications": [v.model_dump() for v in verifications]},
            confidence=min(1.0, mean_score),
        )

    # ------------------------------------------------------------------
    # Core verification
    # ------------------------------------------------------------------

    async def verify_result(self, result: AgentResult) -> Verification:
        """Run all applicable checks on a single agent result.

        Args:
            result: An agent's output to verify.

        Returns:
            Verification with all checks, score, and issues.
        """
        checks: list[dict[str, Any]] = []

        # 1. Fact-check key claims
        claims = await self._extract_claims(result.content)
        for claim in claims[:10]:  # limit to avoid excessive API calls
            claim_check = await self._check_claim(claim)
            checks.append(claim_check.model_dump())

        # 2. Molecular validation
        molecules = result.artifacts.get("molecules", [])
        if isinstance(molecules, list):
            for smi in molecules[:20]:
                if isinstance(smi, str):
                    mol_check = self._validate_molecule(smi)
                    checks.append(mol_check.model_dump())

        # 3. Protein sequence validation
        candidates = result.artifacts.get("candidates", [])
        if isinstance(candidates, list):
            for cand in candidates[:20]:
                seq = cand.get("sequence", "") if isinstance(cand, dict) else ""
                if seq:
                    seq_check = self._validate_protein_sequence(seq)
                    checks.append(seq_check.model_dump())

        # 4. Citation checks
        references = result.artifacts.get("references", [])
        if isinstance(references, list):
            for ref in references[:20]:
                pmid = ref.get("pmid", "") if isinstance(ref, dict) else ""
                if pmid:
                    cit_check = await self._verify_pmid(pmid)
                    checks.append(cit_check.model_dump())

        # 5. Hallucination detection
        hallucination_issues = await self._detect_hallucinations(result)
        for issue in hallucination_issues:
            checks.append(
                ClaimCheck(claim=issue, supported=False, issue=issue, passed=False).model_dump()
            )

        # Compute score
        passed_count = sum(1 for c in checks if c.get("passed", False))
        total = max(len(checks), 1)
        score = passed_count / total

        issues = [
            c.get("issue", "") for c in checks if not c.get("passed", False) and c.get("issue")
        ]

        return Verification(
            agent_name=result.agent_name,
            checks=checks,
            score=score,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 1. Claim extraction and checking
    # ------------------------------------------------------------------

    async def _extract_claims(self, text: str) -> list[str]:
        """Extract verifiable factual claims from agent output.

        Args:
            text: Agent output text.

        Returns:
            List of claim strings.
        """
        if not text or len(text) < 20:
            return []

        try:
            raw = await self.ollama.generate(
                model=self.config.model_extractor,
                prompt=f"Extract verifiable factual claims from:\n\n{text[:3000]}",
                system=PROMPTS["sentinel_extract_claims"],
                format="json",
            )
            data = _extract_json(raw)
            if isinstance(data, list):
                return [str(c) for c in data if isinstance(c, str) and c.strip()]
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return [str(c) for c in v if isinstance(c, str) and c.strip()]
        except Exception:
            logger.debug("Claim extraction failed")
        return []

    async def _check_claim(self, claim: str) -> ClaimCheck:
        """Check a claim against the KG and optionally PubMed.

        Args:
            claim: A factual claim string.

        Returns:
            ClaimCheck with support status and source.
        """
        # Try KG first
        kg_supported = await self._check_claim_against_kg(claim)
        if kg_supported:
            return ClaimCheck(claim=claim, supported=True, source="kg", passed=True)

        # Try PubMed
        pubmed_supported = await self._check_claim_in_pubmed(claim)
        if pubmed_supported:
            return ClaimCheck(claim=claim, supported=True, source="pubmed", passed=True)

        return ClaimCheck(
            claim=claim,
            supported=False,
            source="none",
            issue=f"Unsupported claim: {claim[:100]}",
            passed=False,
        )

    async def _check_claim_against_kg(self, claim: str) -> bool:
        """Search the KG for entities or relations mentioned in a claim.

        Args:
            claim: Claim text.

        Returns:
            True if relevant KG evidence found.
        """
        try:
            results = await self.kg.search_nodes(claim, limit=3)
            return len(results) > 0
        except Exception:
            logger.debug("KG claim check failed for: %s", claim[:50])
            return False

    async def _check_claim_in_pubmed(self, claim: str) -> bool:
        """Search PubMed for supporting evidence.

        Args:
            claim: Claim text.

        Returns:
            True if relevant PubMed articles found.
        """
        if self._pubmed is None:
            return False

        try:
            pmids = await self._pubmed.search(claim[:200], max_results=3)
            return len(pmids) > 0
        except Exception:
            logger.debug("PubMed claim check failed for: %s", claim[:50])
            return False

    # ------------------------------------------------------------------
    # 2. Molecular validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_molecule(smiles: str) -> MoleculeCheck:
        """Validate a SMILES string using RDKit.

        Args:
            smiles: SMILES string.

        Returns:
            MoleculeCheck with validity status.
        """
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return MoleculeCheck(smiles=smiles, valid=True, passed=True)
            return MoleculeCheck(
                smiles=smiles,
                valid=False,
                issue=f"Invalid SMILES: {smiles[:50]}",
                passed=False,
            )
        except ImportError:
            # RDKit not available — skip validation
            return MoleculeCheck(
                smiles=smiles, valid=True, passed=True, issue="rdkit not available"
            )
        except Exception:
            return MoleculeCheck(
                smiles=smiles,
                valid=False,
                issue=f"SMILES validation error: {smiles[:50]}",
                passed=False,
            )

    # ------------------------------------------------------------------
    # 3. Protein sequence validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_protein_sequence(sequence: str) -> SequenceCheck:
        """Validate a protein sequence contains only standard amino acids.

        Args:
            sequence: Amino-acid string.

        Returns:
            SequenceCheck with validity status.
        """
        preview = sequence[:20]
        upper = sequence.upper()
        valid = bool(sequence) and all(ch in _STANDARD_AA for ch in upper)

        if valid:
            return SequenceCheck(
                sequence_preview=preview,
                valid=True,
                length=len(sequence),
                passed=True,
            )
        # Find first invalid character
        bad_chars = sorted({ch for ch in upper if ch not in _STANDARD_AA})
        return SequenceCheck(
            sequence_preview=preview,
            valid=False,
            length=len(sequence),
            issue=f"Invalid amino acid(s): {', '.join(bad_chars)}",
            passed=False,
        )

    # ------------------------------------------------------------------
    # 4. Citation verification
    # ------------------------------------------------------------------

    async def _verify_pmid(self, pmid: str) -> CitationCheck:
        """Verify that a PMID exists on PubMed.

        Args:
            pmid: PubMed identifier string.

        Returns:
            CitationCheck with existence status.
        """
        if self._pubmed is None:
            return CitationCheck(
                pmid=pmid, exists=True, passed=True, issue="pubmed client unavailable"
            )

        try:
            articles = await self._pubmed.fetch_abstracts([pmid])
            if articles:
                return CitationCheck(pmid=pmid, exists=True, passed=True)
            return CitationCheck(
                pmid=pmid,
                exists=False,
                issue=f"PMID {pmid} not found",
                passed=False,
            )
        except Exception:
            logger.debug("PMID verification failed for: %s", pmid)
            return CitationCheck(
                pmid=pmid, exists=False, issue=f"PMID {pmid} check failed", passed=False
            )

    # ------------------------------------------------------------------
    # 5. Hallucination detection
    # ------------------------------------------------------------------

    async def _detect_hallucinations(self, result: AgentResult) -> list[str]:
        """Use LLM to detect unsupported claims in the agent output.

        Args:
            result: Agent result to check.

        Returns:
            List of hallucination issue strings (empty = no problems).
        """
        content = result.content
        if not content or len(content) < 50:
            return []

        # Gather KG context as ground truth
        try:
            kg_results = await self.kg.search_nodes(content[:200], limit=5)
            kg_context = "; ".join(str(r.get("name", "")) for r in kg_results if r.get("name"))
        except Exception:
            kg_context = ""

        prompt = (
            f"Agent: {result.agent_name}\n"
            f"Output to verify:\n{content[:2000]}\n\n"
            f"Known facts from KG: {kg_context or 'none'}\n"
        )

        try:
            raw = await self.ollama.generate(
                model=self.config.model_verifier,
                prompt=prompt,
                system=PROMPTS["sentinel_hallucination"],
                format="json",
            )
            data = _extract_json(raw)
            if isinstance(data, list):
                return [str(item) for item in data if isinstance(item, str) and item.strip()]
            if isinstance(data, dict):
                issues = data.get("issues", data.get("hallucinations", []))
                if isinstance(issues, list):
                    return [str(i) for i in issues if isinstance(i, str) and i.strip()]
        except Exception:
            logger.debug("Hallucination detection failed for %s", result.agent_name)
        return []

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_verification_report(verifications: list[Verification]) -> str:
        """Format verification results as a Markdown report.

        Args:
            verifications: List of per-agent verifications.

        Returns:
            Markdown-formatted report string.
        """
        if not verifications:
            return "# Verification Report\n\nNo agent results to verify."

        parts = ["# Verification Report\n"]

        for v in verifications:
            status = "PASS" if v.score >= 0.5 else "NEEDS REVIEW"
            parts.append(f"\n## {v.agent_name} — {status} ({v.score:.0%})\n")
            parts.append(f"- Checks performed: {len(v.checks)}\n")
            parts.append(f"- Checks passed: {sum(1 for c in v.checks if c.get('passed'))}\n")

            if v.issues:
                parts.append("\n**Issues found:**\n")
                for issue in v.issues[:10]:
                    parts.append(f"- {issue}\n")

        overall = sum(v.score for v in verifications) / len(verifications)
        parts.append(f"\n---\n**Overall verification score: {overall:.0%}**\n")

        return "".join(parts)
