"""Tests for pharos.forecasting.trend_detector — trends, drift, convergence."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pharos.config import Settings
from pharos.forecasting.trend_detector import (
    ConvergenceSignal,
    EntityTrend,
    TrendDetector,
    _compute_acceleration,
    _compute_semantic_drift,
    _compute_velocity,
    _cosine_distance,
)
from pharos.tools.pubmed_tools import PubMedArticle, PubMedClient

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _art(
    pmid: str, abstract: str, year: int = 2024, mesh: list[str] | None = None
) -> PubMedArticle:
    return PubMedArticle(
        pmid=pmid,
        title=f"Article {pmid}",
        abstract=abstract,
        authors=["Author A"],
        journal="J",
        year=year,
        mesh_terms=mesh or [],
    )


# ------------------------------------------------------------------
# Math utility tests
# ------------------------------------------------------------------


class TestComputeVelocity:
    """Test linear trend computation."""

    def test_increasing_trend(self) -> None:
        counts = {2020: 10, 2021: 20, 2022: 30, 2023: 40}
        v = _compute_velocity(counts)
        assert v == pytest.approx(10.0)

    def test_flat_trend(self) -> None:
        counts = {2020: 5, 2021: 5, 2022: 5}
        v = _compute_velocity(counts)
        assert v == pytest.approx(0.0)

    def test_decreasing_trend(self) -> None:
        counts = {2020: 30, 2021: 20, 2022: 10}
        v = _compute_velocity(counts)
        assert v == pytest.approx(-10.0)

    def test_single_year(self) -> None:
        assert _compute_velocity({2020: 10}) == 0.0

    def test_empty(self) -> None:
        assert _compute_velocity({}) == 0.0


class TestComputeAcceleration:
    """Test acceleration (change in velocity)."""

    def test_accelerating(self) -> None:
        # First half flat, second half rising
        counts = {2018: 5, 2019: 5, 2020: 5, 2021: 10, 2022: 20, 2023: 30}
        a = _compute_acceleration(counts)
        assert a > 0

    def test_decelerating(self) -> None:
        counts = {2018: 10, 2019: 20, 2020: 30, 2021: 31, 2022: 32, 2023: 33}
        a = _compute_acceleration(counts)
        assert a < 0

    def test_too_few_points(self) -> None:
        assert _compute_acceleration({2020: 10, 2021: 20}) == 0.0


class TestCosineDistance:
    """Test cosine distance between vectors."""

    def test_identical_vectors(self) -> None:
        assert _cosine_distance([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_distance([1, 0], [-1, 0]) == pytest.approx(2.0)

    def test_zero_vector(self) -> None:
        assert _cosine_distance([0, 0], [1, 1]) == pytest.approx(1.0)


class TestComputeSemanticDrift:
    """Test semantic drift from yearly embeddings."""

    def test_no_drift(self) -> None:
        embs = {2020: [1.0, 0.0], 2021: [1.0, 0.0], 2022: [1.0, 0.0]}
        assert _compute_semantic_drift(embs) == pytest.approx(0.0)

    def test_full_drift(self) -> None:
        embs = {2020: [1.0, 0.0], 2022: [0.0, 1.0]}
        assert _compute_semantic_drift(embs) == pytest.approx(1.0)

    def test_single_year(self) -> None:
        assert _compute_semantic_drift({2020: [1.0, 0.0]}) == 0.0


# ------------------------------------------------------------------
# TrendDetector tests with mocks
# ------------------------------------------------------------------


@pytest.fixture
def mock_pubmed() -> MagicMock:
    client = MagicMock(spec=PubMedClient)
    client.search = AsyncMock(return_value=["1", "2", "3"])
    client.fetch_abstracts = AsyncMock(
        return_value=[
            _art("1", "KRAS mutations drive cancer.", mesh=["Lung Neoplasms", "KRAS"]),
            _art("2", "Sotorasib targets KRAS G12C.", mesh=["Antineoplastic Agents"]),
            _art("3", "Resistance mechanisms in NSCLC.", mesh=["Drug Resistance"]),
        ]
    )
    return client


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.oracle_max_pubmed_queries = 100
    s.oracle_trend_years_start = 2022
    s.oracle_trend_years_end = 2025
    s.embedding_cache_path = ".test_emb_cache"
    return s


@pytest.fixture
def detector(mock_pubmed: MagicMock, mock_ollama: MagicMock, settings: Settings) -> TrendDetector:
    mock_ollama.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    # Disable disk cache for tests
    with (
        patch.object(TrendDetector, "_cache_get", return_value=None),
        patch.object(TrendDetector, "_cache_put"),
    ):
        det = TrendDetector(mock_pubmed, mock_ollama, settings)
    return det


class TestTrendDetectorComputeEntityTrends:
    """Test entity trend computation."""

    async def test_computes_trends(self, detector: TrendDetector, mock_pubmed: MagicMock) -> None:
        with (
            patch.object(detector, "_cache_get", return_value=None),
            patch.object(detector, "_cache_put"),
        ):
            trend = await detector.compute_entity_trends("KRAS")

        assert trend.entity == "KRAS"
        assert len(trend.yearly_counts) > 0
        # Each year should have count of 3 (our mock returns 3 PMIDs)
        for count in trend.yearly_counts.values():
            assert count == 3

    async def test_respects_query_limit(
        self, mock_pubmed: MagicMock, mock_ollama: MagicMock
    ) -> None:
        mock_ollama.embed = AsyncMock(return_value=[0.1, 0.2])
        settings = Settings()
        settings.oracle_max_pubmed_queries = 3
        settings.oracle_trend_years_start = 2020
        settings.oracle_trend_years_end = 2030
        det = TrendDetector(mock_pubmed, mock_ollama, settings)

        with patch.object(det, "_cache_get", return_value=None), patch.object(det, "_cache_put"):
            trend = await det.compute_entity_trends("KRAS")

        # Should stop early due to query limit
        assert len(trend.yearly_counts) < 10

    async def test_handles_empty_search_results(self, mock_ollama: MagicMock) -> None:
        empty_pubmed = MagicMock(spec=PubMedClient)
        empty_pubmed.search = AsyncMock(return_value=[])
        empty_pubmed.fetch_abstracts = AsyncMock(return_value=[])

        settings = Settings()
        settings.oracle_max_pubmed_queries = 100
        settings.oracle_trend_years_start = 2023
        settings.oracle_trend_years_end = 2025
        det = TrendDetector(empty_pubmed, mock_ollama, settings)

        with patch.object(det, "_cache_get", return_value=None), patch.object(det, "_cache_put"):
            trend = await det.compute_entity_trends("nonexistent_xyz")

        assert trend.velocity == 0.0
        for count in trend.yearly_counts.values():
            assert count == 0


class TestTrendDetectorConvergence:
    """Test entity convergence detection."""

    async def test_convergence_detected(self, mock_ollama: MagicMock) -> None:
        async def search_side_effect(query: str, max_results: int = 10) -> list[str]:
            return ["1"]

        async def fetch_side_effect(pmids: list[str]) -> list[PubMedArticle]:
            return [_art("1", "test abstract")]

        pubmed = MagicMock(spec=PubMedClient)
        pubmed.search = AsyncMock(side_effect=search_side_effect)
        pubmed.fetch_abstracts = AsyncMock(side_effect=fetch_side_effect)

        # Embeddings that converge over time
        embeddings = iter(
            [
                [1.0, 0.0, 0.0],  # entity_a 2022
                [0.0, 1.0, 0.0],  # entity_b 2022
                [0.7, 0.3, 0.0],  # entity_a 2023
                [0.3, 0.7, 0.0],  # entity_b 2023
                [0.5, 0.5, 0.0],  # entity_a 2024
                [0.5, 0.5, 0.0],  # entity_b 2024
            ]
        )
        mock_ollama.embed = AsyncMock(side_effect=lambda m, t: next(embeddings))

        settings = Settings()
        settings.oracle_max_pubmed_queries = 100
        settings.oracle_trend_years_start = 2022
        settings.oracle_trend_years_end = 2025
        det = TrendDetector(pubmed, mock_ollama, settings)

        with patch.object(det, "_cache_get", return_value=None), patch.object(det, "_cache_put"):
            signal = await det.find_converging_entities("entityA", "entityB")

        assert signal.convergence_rate > 0
        assert signal.entity_a == "entityA"
        assert signal.entity_b == "entityB"

    async def test_no_convergence_on_empty(self, mock_ollama: MagicMock) -> None:
        pubmed = MagicMock(spec=PubMedClient)
        pubmed.search = AsyncMock(return_value=[])
        pubmed.fetch_abstracts = AsyncMock(return_value=[])

        settings = Settings()
        settings.oracle_max_pubmed_queries = 100
        settings.oracle_trend_years_start = 2023
        settings.oracle_trend_years_end = 2025
        det = TrendDetector(pubmed, mock_ollama, settings)

        with patch.object(det, "_cache_get", return_value=None), patch.object(det, "_cache_put"):
            signal = await det.find_converging_entities("A", "B")

        assert signal.convergence_rate == 0.0


# ------------------------------------------------------------------
# Model tests
# ------------------------------------------------------------------


class TestEntityTrendModel:
    def test_defaults(self) -> None:
        t = EntityTrend(entity="TP53")
        assert t.velocity == 0.0
        assert t.yearly_counts == {}
        assert t.emerging_associations == []


class TestConvergenceSignalModel:
    def test_defaults(self) -> None:
        s = ConvergenceSignal(entity_a="A", entity_b="B")
        assert s.convergence_rate == 0.0
        assert s.years_to_convergence is None
