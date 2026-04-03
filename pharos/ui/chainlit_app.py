"""Chainlit UI — interactive chat interface for PHAROS.

Run with: ``chainlit run pharos/ui/chainlit_app.py``

Features:
- Chat interface with Markdown rendering.
- Sidebar with KG stats, loaded models, and agent activity log.
- File upload (PDF → Scribe, PDB → Architect).
- Streaming partial results during agent execution.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# Ensure project root is on sys.path when run via chainlit CLI
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chainlit as cl

from pharos.agents.architect import ArchitectAgent
from pharos.agents.general import GeneralAgent
from pharos.agents.cartographer import CartographerAgent
from pharos.agents.oracle import OracleAgent
from pharos.agents.router import RouterAgent
from pharos.agents.scribe import ScribeAgent
from pharos.agents.sentinel import SentinelAgent
from pharos.config import get_settings
from pharos.graph.entity_extractor import BioEntityExtractor
from pharos.graph.neo4j_manager import Neo4jManager
from pharos.orchestration.graph_workflow import build_workflow
from pharos.orchestration.task_models import Task, TaskType, WorkflowState
from pharos.tools.ollama_client import OllamaClient
from pharos.tools.pubmed_tools import PubMedClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_agents(
    ollama: OllamaClient,
    kg: Neo4jManager,
    config: Any,
) -> dict[str, Any]:
    """Instantiate all PHAROS agents.

    Args:
        ollama: Shared Ollama client.
        kg: Shared Neo4j manager.
        config: Application settings.

    Returns:
        Dict mapping agent names to agent instances.
    """
    pubmed = PubMedClient(config)
    extractor = BioEntityExtractor(ollama, config)

    return {
        "router": RouterAgent(ollama, kg, config),
        "oracle": OracleAgent(ollama, kg, config, pubmed=pubmed),
        "scribe": ScribeAgent(ollama, kg, config, pubmed=pubmed),
        "cartographer": CartographerAgent(ollama, kg, config, pubmed=pubmed, extractor=extractor),
        "alchemist": __import__(
            "pharos.agents.alchemist", fromlist=["AlchemistAgent"]
        ).AlchemistAgent(ollama, kg, config),
        "architect": ArchitectAgent(ollama, kg, config),
        "sentinel": SentinelAgent(ollama, kg, config, pubmed=pubmed),
        "general": GeneralAgent(ollama, kg, config),
    }


async def _get_kg_stats(kg: Neo4jManager) -> str:
    """Fetch KG statistics for the sidebar.

    Args:
        kg: Neo4j manager.

    Returns:
        Formatted stats string.
    """
    try:
        stats = await kg.stats()
        total_nodes = sum(stats.values())
        lines = ["**Knowledge Graph**", f"- Total nodes: {total_nodes}"]
        for label, count in sorted(stats.items()):
            if count > 0:
                lines.append(f"- {label}: {count}")
        return "\n".join(lines)
    except Exception:
        return "**Knowledge Graph**: unavailable"


def _detect_file_task_type(filename: str) -> TaskType | None:
    """Detect task type from uploaded file extension.

    Args:
        filename: Name of the uploaded file.

    Returns:
        TaskType if recognized, else None.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return TaskType.REVIEW
    if ext == ".pdb":
        return TaskType.DESIGN_PROTEIN
    return None


# ---------------------------------------------------------------------------
# Chainlit event handlers
# ---------------------------------------------------------------------------


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize session resources when a user connects."""
    settings = get_settings()
    ollama = OllamaClient(settings)

    alive = await ollama.is_alive()
    if not alive:
        await cl.Message(
            content="Ollama is not reachable. Please start Ollama and refresh."
        ).send()
        return

    kg = Neo4jManager(settings)

    cl.user_session.set("settings", settings)
    cl.user_session.set("ollama", ollama)
    cl.user_session.set("kg", kg)
    cl.user_session.set("activity_log", [])
    # Build agents and workflow
    agents = _build_agents(ollama, kg, settings)
    workflow = build_workflow(agents)
    cl.user_session.set("workflow", workflow)
    cl.user_session.set("agents", agents)
    # Show welcome with sidebar info
    kg_stats = await _get_kg_stats(kg)

    models_info = "**Loaded models**: checking…"
    try:
        model_list = await ollama.list_models()
        if model_list:
            model_names = [m.get("name", "?") for m in model_list[:10]]
            models_info = f"**Available models**: {', '.join(model_names)}"
        else:
            models_info = "**Available models**: none found"
    except Exception:
        models_info = "**Available models**: error fetching"

    welcome = (
        "Welcome to **PHAROS** — Platform for Hypothesis-driven "
        "Autonomous Research, Orchestration & Synthesis.\n\n"
        "Ask me any biomedical research question, or upload a file:\n"
        "- **PDF** → literature review (Scribe)\n"
        "- **PDB** → protein design (Architect)\n\n"
        f"---\n{kg_stats}\n\n{models_info}"
    )

    await cl.Message(content=welcome).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle an incoming user message."""
    settings = cl.user_session.get("settings")
    workflow = cl.user_session.get("workflow")
    activity_log: list[str] = cl.user_session.get("activity_log") or []
    if settings is None or workflow is None:
        await cl.Message(content="Session not initialized. Please refresh.").send()
        return

    # Check for file uploads
    task_type = None
    file_context = ""
    if message.elements:
        for element in message.elements:
            if hasattr(element, "name") and hasattr(element, "path"):
                detected = _detect_file_task_type(element.name)
                if detected:
                    task_type = detected
                    file_context = f"\n[Uploaded file: {element.name} at {element.path}]"
                    break

    query = message.content + file_context
    task = Task(query=query, task_type=task_type)

    # Send progress indicator
    progress_msg = cl.Message(content="**Processing…** Routing your query to the right agent.")
    await progress_msg.send()

    activity_log.append(f"Query: {query[:100]}")

    # Build initial state
    state = WorkflowState(
        task=task,
        results=[],
        current_agent="router",
        kg_context="",
        iteration=0,
    )

    # Execute workflow
    try:
        final_state: dict[str, Any] = {}
        async for event in workflow.astream(state):
            if isinstance(event, dict):
                for _node_name, node_state in event.items():
                    final_state = node_state if isinstance(node_state, dict) else {}
                    if isinstance(node_state, dict) and "results" in node_state:
                        results = node_state["results"]
                        if results:
                            last = results[-1]
                            activity_log.append(
                                f"{last.agent_name}: confidence={last.confidence:.0%}"
                            )

        # Final output from last state
        final_output = final_state.get("kg_context", "No output generated.")
        if not final_output or final_output.strip() == "# PHAROS Results":
            # Fallback: grab content from results directly
            all_results = final_state.get("results", [])
            for r in all_results:
                if r.agent_name.lower() not in ("router", "sentinel") and r.content:
                    final_output = r.content
                    break
        await progress_msg.remove()
        await cl.Message(content=final_output).send()

    except Exception:
        logger.exception("Workflow execution failed")
        await progress_msg.remove()
        await cl.Message(
            content="An error occurred during processing. Please check the logs and try again."
        ).send()

    cl.user_session.set("activity_log", activity_log)
