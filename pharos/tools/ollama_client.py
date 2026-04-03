"""Async wrapper for the Ollama HTTP API with retry, streaming, and model management."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pharos.config import Settings

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when the Ollama API returns an error."""


class OllamaClient:
    """Async Ollama API client built on httpx.

    Provides generation, chat, embedding, streaming, health checks,
    and model management with automatic retry and exponential backoff.

    Args:
        config: Application settings (provides host URL, retry params).
    """

    def __init__(self, config: Settings | None = None) -> None:
        from pharos.config import get_settings

        self._config = config or get_settings()
        self._base_url = self._config.ollama_host
        self._max_retries = self._config.max_retries
        self._base_delay = self._config.retry_base_delay
        self._timeout = self._config.request_timeout

    def _client(self) -> httpx.AsyncClient:
        """Create a new httpx async client."""
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Send an HTTP request with exponential-backoff retry.

        Args:
            method: HTTP method (GET, POST).
            path: API endpoint path.
            json_body: Optional JSON request body.

        Returns:
            The httpx Response object.

        Raises:
            OllamaError: If all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._client() as client:
                    response = await client.request(method, path, json=json_body)
                    response.raise_for_status()
                    return response
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._base_delay * (2**attempt)
                    logger.warning(
                        "Ollama request failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        raise OllamaError(
            f"Ollama request to {path} failed after {self._max_retries + 1} attempts"
        ) from last_exc

    # --- Generation ------------------------------------------------------

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        format: str = "",  # noqa: A002
    ) -> str:
        """Generate a completion (non-streaming).

        Args:
            model: Ollama model name (e.g. "llama3.2:3b").
            prompt: User prompt.
            system: Optional system prompt.
            format: Optional response format ("json" for JSON mode).

        Returns:
            The generated text.
        """
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            body["system"] = system
        if format:
            body["format"] = format

        response = await self._request_with_retry("POST", "/api/generate", body)
        data: dict[str, Any] = response.json()
        return str(data.get("response", ""))

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: str = "",
    ) -> AsyncIterator[str]:
        """Generate a completion with streaming.

        Args:
            model: Ollama model name.
            prompt: User prompt.
            system: Optional system prompt.

        Yields:
            Text chunks as they arrive.
        """
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            body["system"] = system

        async with (
            self._client() as client,
            client.stream("POST", "/api/generate", json=body) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json as _json

                chunk: dict[str, Any] = _json.loads(line)
                text = chunk.get("response", "")
                if text:
                    yield text

    # --- Chat ------------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        format: str = "",  # noqa: A002
    ) -> str:
        """Send a multi-turn chat completion.

        Args:
            model: Ollama model name.
            messages: List of message dicts with 'role' and 'content'.
            format: Optional response format ("json" for JSON mode).

        Returns:
            The assistant's response text.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if format:
            body["format"] = format

        response = await self._request_with_retry("POST", "/api/chat", body)
        data: dict[str, Any] = response.json()
        message: dict[str, Any] = data.get("message", {})
        return str(message.get("content", ""))

    # --- Embeddings ------------------------------------------------------

    async def embed(self, model: str, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            model: Embedding model name (e.g. "all-minilm:l6-v2").
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        body: dict[str, Any] = {
            "model": model,
            "input": text,
        }
        response = await self._request_with_retry("POST", "/api/embed", body)
        data: dict[str, Any] = response.json()
        embeddings: list[list[float]] = data.get("embeddings", [[]])
        return embeddings[0] if embeddings else []

    # --- Health & model management ---------------------------------------

    async def is_alive(self) -> bool:
        """Check if the Ollama server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        try:
            async with self._client() as client:
                response = await client.get("/")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List all locally available models.

        Returns:
            List of model info dicts from the Ollama API.
        """
        response = await self._request_with_retry("GET", "/api/tags")
        data: dict[str, Any] = response.json()
        models: list[dict[str, Any]] = data.get("models", [])
        return models

    async def pull_model(self, name: str) -> None:
        """Pull (download) a model by name.

        Args:
            name: Model name to pull (e.g. "llama3.2:3b").
        """
        logger.info("Pulling model %s ...", name)
        body: dict[str, Any] = {"name": name, "stream": False}
        await self._request_with_retry("POST", "/api/pull", body)
        logger.info("Model %s pulled successfully", name)
