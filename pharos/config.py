"""Application configuration via Pydantic Settings.

Loads from environment variables and .env file. All settings have sensible
defaults for local development with Ollama + Neo4j.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global PHAROS configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PHAROS_",
        case_sensitive=False,
    )

    # --- Ollama ----------------------------------------------------------
    ollama_host: str = "http://localhost:11434"

    # --- Neo4j -----------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "pharos-dev"

    # --- Model mapping ---------------------------------------------------
    model_router: str = "llama3.2:3b"
    model_extractor: str = "llama3.1:8b"
    model_verifier: str = "phi4:14b"
    model_reasoner: str = "llama3.3:70b"
    model_coder: str = "qwen2.5-coder:32b"
    embedding_model: str = "all-minilm:l6-v2"

    # --- PubMed / NCBI ---------------------------------------------------
    ncbi_api_key: str = ""
    pubmed_rate_limit: float = 3.0  # requests/sec (10.0 with API key)

    # --- Oracle / forecasting --------------------------------------------
    embedding_cache_path: str = ".pharos_embedding_cache"
    oracle_max_pubmed_queries: int = 50
    oracle_trend_years_start: int = 2015
    oracle_trend_years_end: int = 2027

    # --- Architect / protein design ---------------------------------------
    proteinmpnn_path: str = ""
    architect_max_candidates: int = 10
    architect_esmfold_timeout: int = 600

    # --- Runtime ---------------------------------------------------------
    max_retries: int = 3
    retry_base_delay: float = 1.0
    request_timeout: float = 120.0
    max_workflow_iterations: int = 10


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
