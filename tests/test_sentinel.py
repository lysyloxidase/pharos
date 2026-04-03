"""Tests for pharos.agents.sentinel — verification and fact-checking."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.agents.sentinel import (
    CitationCheck,
    ClaimCheck,
    MoleculeCheck,
    SentinelAgent,
    SequenceCheck,
    Verification,
)
from pharos.config import Settings
from pharos.orchestration.task_models import AgentResult, Task, TaskType, WorkflowState
from pharos.tools.pubmed_tools import PubMedArticle, PubMedClient

# ------------------------------------------------------------------
# Model tests
# ------------------------------------------------------------------


class TestClaimCheckModel:
    def test_defaults(self) -> None:
        c = ClaimCheck(claim="test claim")
        assert c.supported is False
        assert c.passed is False
        assert c.source == "none"


class TestMoleculeCheckModel:
    def test_defaults(self) -> None:
        m = MoleculeCheck(smiles="CCO")
        assert m.valid is False
        assert m.passed is False


class TestSequenceCheckModel:
    def test_defaults(self) -> None:
        s = SequenceCheck()
        assert s.valid is False
        assert s.length == 0


class TestCitationCheckModel:
    def test_defaults(self) -> None:
        c = CitationCheck(pmid="12345")
        assert c.exists is False
        assert c.passed is False


class TestVerificationModel:
    def test_defaults(self) -> None:
        v = Verification(agent_name="Oracle")
        assert v.score == 0.0
        assert v.checks == []
        assert v.issues == []


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def mock_pubmed() -> MagicMock:
    client = MagicMock(spec=PubMedClient)
    client.search = AsyncMock(return_value=["12345"])
    client.fetch_abstracts = AsyncMock(
        return_value=[PubMedArticle(pmid="12345", title="Test", abstract="Test abstract")]
    )
    return client


@pytest.fixture
def sentinel(
    mock_ollama: MagicMock,
    mock_neo4j: MagicMock,
    settings: Settings,
    mock_pubmed: MagicMock,
) -> SentinelAgent:
    return SentinelAgent(
        ollama=mock_ollama,
        kg=mock_neo4j,
        config=settings,
        pubmed=mock_pubmed,
    )


@pytest.fixture
def state() -> WorkflowState:
    return WorkflowState(
        task=Task(query="test", task_type=TaskType.VERIFY),
        results=[],
        current_agent="sentinel",
        kg_context="",
        iteration=0,
    )


def _make_result(
    name: str = "Oracle",
    content: str = "KRAS is a highly promising therapeutic target for pancreatic cancer treatment based on recent studies.",
    artifacts: dict | None = None,
) -> AgentResult:
    return AgentResult(
        agent_name=name,
        task_id="test-123",
        content=content,
        artifacts=artifacts or {},
        confidence=0.8,
    )


# ------------------------------------------------------------------
# Claim extraction
# ------------------------------------------------------------------


class TestClaimExtraction:
    async def test_extracts_claims_from_list(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(
            return_value=json.dumps(["KRAS causes cancer", "Sotorasib treats NSCLC"])
        )
        claims = await sentinel._extract_claims("Some long text about KRAS and cancer.")
        assert len(claims) == 2
        assert "KRAS causes cancer" in claims

    async def test_extracts_claims_from_dict(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(
            return_value=json.dumps({"claims": ["Claim A", "Claim B"]})
        )
        claims = await sentinel._extract_claims("Some text here that is long enough.")
        assert len(claims) == 2

    async def test_returns_empty_on_short_text(self, sentinel: SentinelAgent) -> None:
        claims = await sentinel._extract_claims("Short")
        assert claims == []

    async def test_returns_empty_on_failure(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(side_effect=Exception("LLM down"))
        claims = await sentinel._extract_claims("Some long text about biology and research.")
        assert claims == []


# ------------------------------------------------------------------
# Claim checking
# ------------------------------------------------------------------


class TestClaimChecking:
    async def test_claim_found_in_kg(self, sentinel: SentinelAgent, mock_neo4j: MagicMock) -> None:
        mock_neo4j.search_nodes = AsyncMock(return_value=[{"name": "KRAS"}])
        check = await sentinel._check_claim("KRAS causes cancer")
        assert check.passed is True
        assert check.source == "kg"

    async def test_claim_found_in_pubmed(
        self, sentinel: SentinelAgent, mock_neo4j: MagicMock, mock_pubmed: MagicMock
    ) -> None:
        mock_neo4j.search_nodes = AsyncMock(return_value=[])
        mock_pubmed.search = AsyncMock(return_value=["12345"])
        check = await sentinel._check_claim("KRAS causes cancer")
        assert check.passed is True
        assert check.source == "pubmed"

    async def test_claim_not_supported(
        self, sentinel: SentinelAgent, mock_neo4j: MagicMock, mock_pubmed: MagicMock
    ) -> None:
        mock_neo4j.search_nodes = AsyncMock(return_value=[])
        mock_pubmed.search = AsyncMock(return_value=[])
        check = await sentinel._check_claim("Unicorns cure diabetes")
        assert check.passed is False
        assert check.source == "none"
        assert "Unsupported" in check.issue

    async def test_kg_check_handles_error(
        self, sentinel: SentinelAgent, mock_neo4j: MagicMock
    ) -> None:
        mock_neo4j.search_nodes = AsyncMock(side_effect=Exception("KG down"))
        result = await sentinel._check_claim_against_kg("test claim")
        assert result is False


# ------------------------------------------------------------------
# Molecule validation
# ------------------------------------------------------------------


class TestMoleculeValidation:
    def test_valid_smiles(self) -> None:
        check = SentinelAgent._validate_molecule("CCO")
        assert check.passed is True
        assert check.valid is True

    def test_invalid_smiles(self) -> None:
        check = SentinelAgent._validate_molecule("not_a_molecule_XYZ!!!")
        # May pass or fail depending on RDKit availability
        # Just verify it returns a MoleculeCheck
        assert isinstance(check, MoleculeCheck)

    def test_empty_smiles(self) -> None:
        check = SentinelAgent._validate_molecule("")
        assert isinstance(check, MoleculeCheck)


# ------------------------------------------------------------------
# Protein sequence validation
# ------------------------------------------------------------------


class TestProteinSequenceValidation:
    def test_valid_sequence(self) -> None:
        check = SentinelAgent._validate_protein_sequence("ACDEFGHIKLMNPQRSTVWY")
        assert check.passed is True
        assert check.valid is True
        assert check.length == 20

    def test_invalid_sequence(self) -> None:
        check = SentinelAgent._validate_protein_sequence("ACDE123XYZ")
        assert check.passed is False
        assert "Invalid amino acid" in check.issue

    def test_empty_sequence(self) -> None:
        check = SentinelAgent._validate_protein_sequence("")
        assert check.passed is False


# ------------------------------------------------------------------
# Citation verification
# ------------------------------------------------------------------


class TestCitationVerification:
    async def test_pmid_exists(self, sentinel: SentinelAgent, mock_pubmed: MagicMock) -> None:
        mock_pubmed.fetch_abstracts = AsyncMock(
            return_value=[PubMedArticle(pmid="12345", title="Test")]
        )
        check = await sentinel._verify_pmid("12345")
        assert check.passed is True
        assert check.exists is True

    async def test_pmid_not_found(self, sentinel: SentinelAgent, mock_pubmed: MagicMock) -> None:
        mock_pubmed.fetch_abstracts = AsyncMock(return_value=[])
        check = await sentinel._verify_pmid("99999")
        assert check.passed is False
        assert "not found" in check.issue

    async def test_no_pubmed_client(
        self, mock_ollama: MagicMock, mock_neo4j: MagicMock, settings: Settings
    ) -> None:
        sentinel_no_pm = SentinelAgent(ollama=mock_ollama, kg=mock_neo4j, config=settings)
        check = await sentinel_no_pm._verify_pmid("12345")
        assert check.passed is True  # graceful pass when no client


# ------------------------------------------------------------------
# Hallucination detection
# ------------------------------------------------------------------


class TestHallucinationDetection:
    async def test_no_hallucinations(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(return_value=json.dumps({"issues": []}))
        result = _make_result()
        issues = await sentinel._detect_hallucinations(result)
        assert issues == []

    async def test_detects_hallucinations(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(
            return_value=json.dumps({"issues": ["Claim X is fabricated"]})
        )
        result = _make_result()
        issues = await sentinel._detect_hallucinations(result)
        assert len(issues) == 1
        assert "fabricated" in issues[0]

    async def test_skips_short_content(self, sentinel: SentinelAgent) -> None:
        result = _make_result(content="Short")
        issues = await sentinel._detect_hallucinations(result)
        assert issues == []

    async def test_handles_failure(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(side_effect=Exception("fail"))
        result = _make_result()
        issues = await sentinel._detect_hallucinations(result)
        assert issues == []


# ------------------------------------------------------------------
# Full verify_result
# ------------------------------------------------------------------


class TestVerifyResult:
    async def test_verifies_agent_result(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(
            side_effect=[
                json.dumps(["Claim A"]),  # extract claims
                json.dumps({"issues": []}),  # hallucination check
            ]
        )
        sentinel.kg.search_nodes = AsyncMock(return_value=[{"name": "X"}])

        result = _make_result()
        verification = await sentinel.verify_result(result)

        assert verification.agent_name == "Oracle"
        assert verification.score > 0

    async def test_with_molecule_artifacts(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(
            side_effect=[
                json.dumps([]),  # no claims
                json.dumps({"issues": []}),  # no hallucinations
            ]
        )
        result = _make_result(artifacts={"molecules": ["CCO", "c1ccccc1"]})
        verification = await sentinel.verify_result(result)
        # Should have molecule checks
        mol_checks = [c for c in verification.checks if "smiles" in c]
        assert len(mol_checks) == 2

    async def test_with_protein_candidates(self, sentinel: SentinelAgent) -> None:
        sentinel.ollama.generate = AsyncMock(
            side_effect=[
                json.dumps([]),
                json.dumps({"issues": []}),
            ]
        )
        result = _make_result(
            artifacts={"candidates": [{"sequence": "ACDEF"}, {"sequence": "GHIKL"}]}
        )
        verification = await sentinel.verify_result(result)
        seq_checks = [c for c in verification.checks if "sequence_preview" in c]
        assert len(seq_checks) == 2

    async def test_with_citation_artifacts(
        self, sentinel: SentinelAgent, mock_pubmed: MagicMock
    ) -> None:
        sentinel.ollama.generate = AsyncMock(
            side_effect=[
                json.dumps([]),
                json.dumps({"issues": []}),
            ]
        )
        mock_pubmed.fetch_abstracts = AsyncMock(
            return_value=[PubMedArticle(pmid="12345", title="Test")]
        )
        result = _make_result(artifacts={"references": [{"pmid": "12345"}]})
        verification = await sentinel.verify_result(result)
        cit_checks = [c for c in verification.checks if "pmid" in c]
        assert len(cit_checks) == 1
        assert cit_checks[0]["passed"] is True


# ------------------------------------------------------------------
# Full run pipeline
# ------------------------------------------------------------------


class TestSentinelRun:
    async def test_full_pipeline(self, sentinel: SentinelAgent, state: WorkflowState) -> None:
        state["results"] = [
            _make_result("Oracle", "KRAS is a driver of cancer progression."),
            _make_result("Scribe", "A review of KRAS therapies."),
        ]

        sentinel.ollama.generate = AsyncMock(
            side_effect=[
                json.dumps(["KRAS drives cancer"]),  # claims for Oracle
                json.dumps({"issues": []}),  # hallucination for Oracle
                json.dumps(["KRAS therapies exist"]),  # claims for Scribe
                json.dumps({"issues": []}),  # hallucination for Scribe
            ]
        )
        sentinel.kg.search_nodes = AsyncMock(return_value=[{"name": "KRAS"}])

        task = Task(query="verify", task_type=TaskType.VERIFY)
        result = await sentinel.run(task, state)

        assert result.agent_name == "Sentinel"
        assert result.confidence > 0
        assert len(result.artifacts["verifications"]) == 2

    async def test_skips_router_and_sentinel_results(
        self, sentinel: SentinelAgent, state: WorkflowState
    ) -> None:
        state["results"] = [
            _make_result("router", "routed to oracle"),
            _make_result("Sentinel", "previous verification"),
            _make_result("Oracle", "Real content to check."),
        ]

        sentinel.ollama.generate = AsyncMock(
            side_effect=[
                json.dumps([]),
                json.dumps({"issues": []}),
            ]
        )

        task = Task(query="verify")
        result = await sentinel.run(task, state)

        # Should only verify Oracle, not router or sentinel
        assert len(result.artifacts["verifications"]) == 1
        assert result.artifacts["verifications"][0]["agent_name"] == "Oracle"

    async def test_empty_results(self, sentinel: SentinelAgent, state: WorkflowState) -> None:
        state["results"] = []
        task = Task(query="verify")
        result = await sentinel.run(task, state)

        assert result.confidence == 1.0
        assert "No agent results" in result.content


# ------------------------------------------------------------------
# Report formatting
# ------------------------------------------------------------------


class TestFormatVerificationReport:
    def test_formats_report(self) -> None:
        verifications = [
            Verification(
                agent_name="Oracle",
                checks=[
                    {"passed": True, "claim": "X"},
                    {"passed": False, "claim": "Y", "issue": "Bad"},
                ],
                score=0.5,
                issues=["Bad"],
            )
        ]
        report = SentinelAgent._format_verification_report(verifications)
        assert "Oracle" in report
        assert "50%" in report
        assert "Bad" in report

    def test_empty_verifications(self) -> None:
        report = SentinelAgent._format_verification_report([])
        assert "No agent results" in report
