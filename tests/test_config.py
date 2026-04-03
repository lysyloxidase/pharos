"""Tests for pharos.config — default values and environment overrides."""

from __future__ import annotations

import os
from unittest.mock import patch

from pharos.config import Settings, get_settings


class TestSettingsDefaults:
    """Verify that all settings have correct default values."""

    def _make(self) -> Settings:
        """Create Settings without reading .env file."""
        return Settings(_env_file=None)  # type: ignore[call-arg]

    def test_ollama_host_default(self) -> None:
        assert self._make().ollama_host == "http://localhost:11434"

    def test_neo4j_uri_default(self) -> None:
        assert self._make().neo4j_uri == "bolt://localhost:7687"

    def test_neo4j_user_default(self) -> None:
        assert self._make().neo4j_user == "neo4j"

    def test_model_router_default(self) -> None:
        assert self._make().model_router == "llama3.2:3b"

    def test_model_extractor_default(self) -> None:
        assert self._make().model_extractor == "llama3.1:8b"

    def test_model_verifier_default(self) -> None:
        assert self._make().model_verifier == "phi4:14b"

    def test_model_reasoner_default(self) -> None:
        assert self._make().model_reasoner == "llama3.3:70b"

    def test_embedding_model_default(self) -> None:
        assert self._make().embedding_model == "all-minilm:l6-v2"

    def test_max_retries_default(self) -> None:
        assert self._make().max_retries == 3


class TestSettingsEnvOverride:
    """Verify that environment variables override defaults."""

    def test_ollama_host_override(self) -> None:
        with patch.dict(os.environ, {"PHAROS_OLLAMA_HOST": "http://gpu-server:11434"}):
            s = Settings()
            assert s.ollama_host == "http://gpu-server:11434"

    def test_neo4j_password_override(self) -> None:
        with patch.dict(os.environ, {"PHAROS_NEO4J_PASSWORD": "secret123"}):
            s = Settings()
            assert s.neo4j_password == "secret123"

    def test_model_router_override(self) -> None:
        with patch.dict(os.environ, {"PHAROS_MODEL_ROUTER": "mistral:7b"}):
            s = Settings()
            assert s.model_router == "mistral:7b"


class TestGetSettings:
    """Test the get_settings helper."""

    def test_returns_settings_instance(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)
