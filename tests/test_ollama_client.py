"""Tests for pharos.tools.ollama_client — retry, generation, health check."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pharos.config import Settings
from pharos.tools.ollama_client import OllamaClient, OllamaError


@pytest.fixture
def client() -> OllamaClient:
    """Create an OllamaClient with test settings."""
    settings = Settings()
    settings.max_retries = 2
    settings.retry_base_delay = 0.01  # fast retries for tests
    return OllamaClient(settings)


class TestGenerate:
    """Test the generate method."""

    async def test_generate_returns_response(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "Hello world"}

        mock_http = AsyncMock()
        mock_http.request = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client, "_client", return_value=mock_http):
            result = await client.generate("llama3.2:3b", "test prompt")
            assert result == "Hello world"

    async def test_generate_with_system_prompt(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "classified"}

        mock_http = AsyncMock()
        mock_http.request = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client, "_client", return_value=mock_http):
            result = await client.generate("llama3.2:3b", "query", system="You are a classifier")
            assert result == "classified"
            call_args = mock_http.request.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert body["system"] == "You are a classifier"


class TestRetry:
    """Test exponential backoff retry logic."""

    async def test_retries_on_connection_error(self, client: OllamaClient) -> None:
        mock_http = AsyncMock()
        mock_http.request = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(client, "_client", return_value=mock_http),
            pytest.raises(OllamaError, match="failed after"),
        ):
            await client.generate("llama3.2:3b", "test")

    async def test_succeeds_after_retry(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "ok"}

        mock_http = AsyncMock()
        mock_http.request = AsyncMock(side_effect=[httpx.ConnectError("fail"), mock_response])
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client, "_client", return_value=mock_http):
            result = await client.generate("llama3.2:3b", "test")
            assert result == "ok"


class TestHealthCheck:
    """Test the is_alive health check."""

    async def test_alive_when_reachable(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client, "_client", return_value=mock_http):
            assert await client.is_alive() is True

    async def test_not_alive_when_unreachable(self, client: OllamaClient) -> None:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client, "_client", return_value=mock_http):
            assert await client.is_alive() is False


class TestEmbed:
    """Test the embed method."""

    async def test_returns_embedding_vector(self, client: OllamaClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        mock_http = AsyncMock()
        mock_http.request = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client, "_client", return_value=mock_http):
            result = await client.embed("all-minilm:l6-v2", "test text")
            assert result == [0.1, 0.2, 0.3]
