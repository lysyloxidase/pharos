"""Trend detection via PubMed publication counts and embedding trajectories.

Tracks how biomedical entities evolve in the literature over time by
measuring publication velocity, semantic drift, and entity convergence.
"""

from __future__ import annotations

import hashlib
import logging
import math
import shelve
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pharos.config import Settings
    from pharos.tools.ollama_client import OllamaClient
    from pharos.tools.pubmed_tools import PubMedClient

logger = logging.getLogger(__name__)

_ABSTRACTS_PER_YEAR = 20


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


class EntityTrend(BaseModel):
    """Publication and semantic trend for a biomedical entity.

    Attributes:
        entity: Entity name.
        yearly_counts: Year to publication count mapping.
        velocity: Linear trend slope (publications/year).
        acceleration: Change in velocity over time.
        semantic_drift: Cosine distance between earliest and latest embeddings.
        emerging_associations: Co-occurring entities with signal strength.
    """

    entity: str
    yearly_counts: dict[int, int] = Field(default_factory=dict)
    velocity: float = 0.0
    acceleration: float = 0.0
    semantic_drift: float = 0.0
    emerging_associations: list[tuple[str, float]] = Field(default_factory=list)


class ConvergenceSignal(BaseModel):
    """Measures whether two entities are converging in embedding space.

    Attributes:
        entity_a: First entity name.
        entity_b: Second entity name.
        convergence_rate: Rate of convergence (-1 to 1, positive = converging).
        years_to_convergence: Estimated years until overlap, if converging.
        evidence_snippets: Supporting text fragments.
    """

    entity_a: str
    entity_b: str
    convergence_rate: float = 0.0
    years_to_convergence: float | None = None
    evidence_snippets: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# TrendDetector
# ------------------------------------------------------------------


class TrendDetector:
    """Analyse publication trends and semantic trajectories for entities.

    Uses PubMed search counts and Ollama embeddings to compute velocity,
    acceleration, semantic drift, and inter-entity convergence signals.

    Args:
        pubmed: Async PubMed client.
        ollama: Async Ollama client (for embeddings).
        config: Application settings.
    """

    def __init__(
        self,
        pubmed: PubMedClient,
        ollama: OllamaClient,
        config: Settings,
    ) -> None:
        self._pubmed = pubmed
        self._ollama = ollama
        self._config = config
        self._embedding_model = config.embedding_model
        self._cache_path = config.embedding_cache_path
        self._query_count = 0
        self._max_queries = config.oracle_max_pubmed_queries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compute_entity_trends(
        self,
        entity: str,
        years: range | None = None,
    ) -> EntityTrend:
        """Compute publication trends and semantic drift for an entity.

        Args:
            entity: Biomedical entity name (gene, disease, drug).
            years: Year range to analyse. Defaults to config range.

        Returns:
            EntityTrend with velocity, acceleration, semantic drift,
            and emerging associations.
        """
        if years is None:
            years = range(
                self._config.oracle_trend_years_start,
                self._config.oracle_trend_years_end,
            )

        yearly_counts: dict[int, int] = {}
        yearly_embeddings: dict[int, list[float]] = {}

        for year in years:
            if self._query_count >= self._max_queries:
                logger.warning("Oracle PubMed query limit reached (%d)", self._max_queries)
                break

            query = f"{entity} AND {year}[dp]"
            pmids = await self._pubmed.search(query, max_results=_ABSTRACTS_PER_YEAR)
            self._query_count += 1
            yearly_counts[year] = len(pmids)

            if pmids:
                articles = await self._pubmed.fetch_abstracts(pmids[:_ABSTRACTS_PER_YEAR])
                self._query_count += 1
                texts = [a.abstract for a in articles if a.abstract]
                if texts:
                    emb = await self._mean_embedding(texts)
                    if emb:
                        yearly_embeddings[year] = emb

        velocity = _compute_velocity(yearly_counts)
        acceleration = _compute_acceleration(yearly_counts)
        semantic_drift = _compute_semantic_drift(yearly_embeddings)

        # Emerging associations from recent abstracts
        emerging: list[tuple[str, float]] = []
        recent_years = sorted(yearly_counts.keys())[-3:]
        if recent_years and self._query_count < self._max_queries:
            emerging = await self._find_cooccurrences(entity, recent_years)

        return EntityTrend(
            entity=entity,
            yearly_counts=yearly_counts,
            velocity=velocity,
            acceleration=acceleration,
            semantic_drift=semantic_drift,
            emerging_associations=emerging,
        )

    async def find_converging_entities(
        self,
        entity_a: str,
        entity_b: str,
    ) -> ConvergenceSignal:
        """Check if two entities converge in embedding space over time.

        Args:
            entity_a: First entity name.
            entity_b: Second entity name.

        Returns:
            ConvergenceSignal with rate and estimated convergence time.
        """
        years = range(
            self._config.oracle_trend_years_start,
            self._config.oracle_trend_years_end,
        )

        distances: dict[int, float] = {}
        evidence: list[str] = []

        for year in years:
            if self._query_count >= self._max_queries - 1:
                break

            emb_a = await self._get_entity_year_embedding(entity_a, year)
            emb_b = await self._get_entity_year_embedding(entity_b, year)

            if emb_a and emb_b:
                dist = _cosine_distance(emb_a, emb_b)
                distances[year] = dist

        if len(distances) < 2:
            return ConvergenceSignal(entity_a=entity_a, entity_b=entity_b)

        sorted_years = sorted(distances.keys())
        early_dist = distances[sorted_years[0]]
        late_dist = distances[sorted_years[-1]]
        span = sorted_years[-1] - sorted_years[0]

        convergence_rate = 0.0 if span == 0 else (early_dist - late_dist) / max(span, 1)

        years_to_conv: float | None = None
        if convergence_rate > 0 and late_dist > 0:
            years_to_conv = late_dist / convergence_rate

        if convergence_rate > 0.01:
            evidence.append(
                f"Embedding distance decreased from {early_dist:.3f} to "
                f"{late_dist:.3f} over {span} years"
            )

        return ConvergenceSignal(
            entity_a=entity_a,
            entity_b=entity_b,
            convergence_rate=max(-1.0, min(1.0, convergence_rate)),
            years_to_convergence=years_to_conv,
            evidence_snippets=evidence,
        )

    def reset_query_count(self) -> None:
        """Reset the PubMed query counter."""
        self._query_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_entity_year_embedding(self, entity: str, year: int) -> list[float] | None:
        """Get or compute the mean embedding for an entity in a given year.

        Args:
            entity: Entity name.
            year: Publication year.

        Returns:
            Mean embedding vector or None.
        """
        cache_key = f"ey:{entity}:{year}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if self._query_count >= self._max_queries:
            return None

        query = f"{entity} AND {year}[dp]"
        pmids = await self._pubmed.search(query, max_results=10)
        self._query_count += 1

        if not pmids:
            return None

        articles = await self._pubmed.fetch_abstracts(pmids[:10])
        self._query_count += 1

        texts = [a.abstract for a in articles if a.abstract]
        if not texts:
            return None

        emb = await self._mean_embedding(texts)
        if emb:
            self._cache_put(cache_key, emb)
        return emb

    async def _mean_embedding(self, texts: list[str]) -> list[float] | None:
        """Compute mean embedding for a list of texts.

        Uses caching to avoid recomputing embeddings for identical texts.

        Args:
            texts: List of text strings.

        Returns:
            Mean embedding vector or None.
        """
        embeddings: list[list[float]] = []
        for text in texts:
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            cache_key = f"emb:{text_hash}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                embeddings.append(cached)
            else:
                emb = await self._ollama.embed(self._embedding_model, text)
                if emb:
                    embeddings.append(emb)
                    self._cache_put(cache_key, emb)

        if not embeddings:
            return None

        dim = len(embeddings[0])
        mean = [0.0] * dim
        for emb in embeddings:
            for i in range(min(dim, len(emb))):
                mean[i] += emb[i]
        n = len(embeddings)
        return [v / n for v in mean]

    async def _find_cooccurrences(
        self, entity: str, recent_years: list[int]
    ) -> list[tuple[str, float]]:
        """Find entities that frequently co-occur with the target recently.

        Uses entity extraction from abstract titles as a lightweight proxy.

        Args:
            entity: Target entity name.
            recent_years: Years to consider.

        Returns:
            List of (co-occurring entity, normalized frequency) tuples.
        """
        cooccur: dict[str, int] = {}
        total = 0

        for year in recent_years:
            if self._query_count >= self._max_queries:
                break
            query = f"{entity} AND {year}[dp]"
            pmids = await self._pubmed.search(query, max_results=10)
            self._query_count += 1

            if not pmids:
                continue

            articles = await self._pubmed.fetch_abstracts(pmids[:10])
            self._query_count += 1

            for art in articles:
                for term in art.mesh_terms:
                    normalized = term.lower().strip()
                    if normalized != entity.lower().strip():
                        cooccur[normalized] = cooccur.get(normalized, 0) + 1
                        total += 1

        if not total:
            return []

        ranked = sorted(cooccur.items(), key=lambda x: x[1], reverse=True)[:10]
        return [(name, count / total) for name, count in ranked]

    # ------------------------------------------------------------------
    # Disk cache (shelve)
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> list[float] | None:
        """Read from the embedding cache.

        Args:
            key: Cache key string.

        Returns:
            Cached embedding vector or None.
        """
        try:
            with shelve.open(self._cache_path) as db:
                val: list[float] | None = db.get(key)
                return val
        except Exception:
            return None

    def _cache_put(self, key: str, value: list[float]) -> None:
        """Write to the embedding cache.

        Args:
            key: Cache key string.
            value: Embedding vector to cache.
        """
        try:
            with shelve.open(self._cache_path) as db:
                db[key] = value
        except Exception:
            logger.warning("Failed to write embedding cache for key %s", key)


# ------------------------------------------------------------------
# Math utilities
# ------------------------------------------------------------------


def _compute_velocity(yearly_counts: dict[int, int]) -> float:
    """Compute linear trend slope (publications per year).

    Args:
        yearly_counts: Year to publication count mapping.

    Returns:
        Slope of the linear fit (positive = growing).
    """
    if len(yearly_counts) < 2:
        return 0.0

    years = sorted(yearly_counts.keys())
    n = len(years)
    sum_x = sum(years)
    sum_y = sum(yearly_counts[y] for y in years)
    sum_xy = sum(y * yearly_counts[y] for y in years)
    sum_x2 = sum(y * y for y in years)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0

    return (n * sum_xy - sum_x * sum_y) / denom


def _compute_acceleration(yearly_counts: dict[int, int]) -> float:
    """Compute acceleration (change in velocity over time).

    Splits the timeline in half and compares velocities.

    Args:
        yearly_counts: Year to publication count mapping.

    Returns:
        Difference between second-half and first-half velocities.
    """
    if len(yearly_counts) < 4:
        return 0.0

    years = sorted(yearly_counts.keys())
    mid = len(years) // 2
    first_half = {y: yearly_counts[y] for y in years[:mid]}
    second_half = {y: yearly_counts[y] for y in years[mid:]}

    v1 = _compute_velocity(first_half)
    v2 = _compute_velocity(second_half)
    return v2 - v1


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine distance (0 = identical, 2 = opposite).
    """
    dim = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(dim))
    norm_a = math.sqrt(sum(x * x for x in a[:dim]))
    norm_b = math.sqrt(sum(x * x for x in b[:dim]))

    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = dot / (norm_a * norm_b)
    return 1.0 - max(-1.0, min(1.0, similarity))


def _compute_semantic_drift(
    yearly_embeddings: dict[int, list[float]],
) -> float:
    """Compute semantic drift between earliest and latest year embeddings.

    Args:
        yearly_embeddings: Year to mean embedding mapping.

    Returns:
        Cosine distance between earliest and latest embeddings (0-2).
    """
    if len(yearly_embeddings) < 2:
        return 0.0

    years = sorted(yearly_embeddings.keys())
    return _cosine_distance(yearly_embeddings[years[0]], yearly_embeddings[years[-1]])
