"""Shared test fixtures — mock Ollama, mock Neo4j, sample tasks."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pharos.config import Settings
from pharos.graph.neo4j_manager import Neo4jManager
from pharos.orchestration.task_models import AgentResult, Task, TaskType, WorkflowState
from pharos.tools.ollama_client import OllamaClient


@pytest.fixture
def settings() -> Settings:
    """Return a test Settings instance with defaults."""
    return Settings()


@pytest.fixture
def mock_ollama() -> OllamaClient:
    """Return a mocked OllamaClient."""
    client = MagicMock(spec=OllamaClient)
    client.generate = AsyncMock(return_value='{"task_type": "general"}')
    client.chat = AsyncMock(return_value="Test response")
    client.embed = AsyncMock(return_value=[0.1] * 384)
    client.is_alive = AsyncMock(return_value=True)
    client.list_models = AsyncMock(return_value=[])
    client.pull_model = AsyncMock()
    return client


@pytest.fixture
def mock_neo4j() -> Neo4jManager:
    """Return a mocked Neo4jManager."""
    manager = MagicMock(spec=Neo4jManager)
    manager.setup_schema = AsyncMock()
    manager.add_node = AsyncMock(return_value="test_node")
    manager.add_relation = AsyncMock()
    manager.query = AsyncMock(return_value=[])
    manager.get_neighbors = AsyncMock(return_value=[])
    manager.search_nodes = AsyncMock(return_value=[])
    manager.stats = AsyncMock(return_value={"Gene": 0, "Disease": 0})
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)
    return manager


@pytest.fixture
def sample_task() -> Task:
    """Return a sample forecast task."""
    return Task(
        query="Will CRISPR therapies for sickle cell be approved by 2026?",
        task_type=TaskType.FORECAST,
    )


@pytest.fixture
def sample_result() -> AgentResult:
    """Return a sample agent result."""
    return AgentResult(
        agent_name="Oracle",
        task_id="test-123",
        content="Forecast: likely approval by 2026",
        confidence=0.85,
    )


@pytest.fixture
def sample_state(sample_task: Task) -> WorkflowState:
    """Return a sample workflow state."""
    return WorkflowState(
        task=sample_task,
        results=[],
        current_agent="router",
        kg_context="",
        iteration=0,
    )


@pytest.fixture
def mock_httpx_response() -> MagicMock:
    """Return a mock httpx.Response."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = {"response": "test output"}
    return response


@pytest.fixture
def mock_httpx_client(mock_httpx_response: MagicMock) -> Any:
    """Return a mock httpx.AsyncClient as async context manager."""
    client = AsyncMock()
    client.request = AsyncMock(return_value=mock_httpx_response)
    client.get = AsyncMock(return_value=mock_httpx_response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client
