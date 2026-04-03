#!/usr/bin/env python3
"""Interactive PHAROS demo — smoke-tests all agents with sample queries.

Usage::

    python scripts/demo.py

The script checks that Ollama and Neo4j are reachable, then runs one
sample query through each implemented agent and prints results.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Ensure the project root is on sys.path when running as a script
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Colour helpers (ANSI)
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def _header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}\n")


def _section(msg: str) -> None:
    print(f"\n{BOLD}{YELLOW}--- {msg} ---{RESET}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


async def check_ollama() -> bool:
    """Verify Ollama is reachable and list available models."""
    from pharos.config import get_settings
    from pharos.tools.ollama_client import OllamaClient

    settings = get_settings()
    client = OllamaClient(settings)

    _section("Checking Ollama")
    alive = await client.is_alive()
    if not alive:
        _fail(f"Ollama not reachable at {settings.ollama_host}")
        return False
    _ok(f"Ollama running at {settings.ollama_host}")

    try:
        models = await client.list_models()
        if models:
            names = [m.get("name", "?") for m in models[:15]]
            print(f"    Available models: {', '.join(names)}")
        else:
            print("    No models found — run: bash scripts/setup_ollama.sh")
    except Exception:
        print("    Could not list models")

    return True


async def check_neo4j() -> bool:
    """Verify Neo4j is reachable."""
    from pharos.config import get_settings
    from pharos.graph.neo4j_manager import Neo4jManager

    settings = get_settings()
    _section("Checking Neo4j")

    try:
        async with Neo4jManager(settings) as kg:
            stats = await kg.stats()
            total = sum(stats.values())
            _ok(f"Neo4j at {settings.neo4j_uri} — {total} nodes in KG")
            return True
    except Exception as exc:
        _fail(f"Neo4j not reachable at {settings.neo4j_uri}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Agent demos
# ---------------------------------------------------------------------------


async def demo_router() -> None:
    """Demo the Router agent."""
    from pharos.agents.router import RouterAgent
    from pharos.config import get_settings
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.orchestration.task_models import Task, WorkflowState
    from pharos.tools.ollama_client import OllamaClient

    _section("Router Agent")
    settings = get_settings()
    ollama = OllamaClient(settings)

    async with Neo4jManager(settings) as kg:
        router = RouterAgent(ollama, kg, settings)
        queries = [
            "Predict KRAS targets in lung cancer",
            "Write a review on CAR-T therapy",
            "Design a nanobody targeting PD-L1",
        ]
        for q in queries:
            task = Task(query=q)
            state = WorkflowState(
                task=task,
                results=[],
                current_agent="router",
                kg_context="",
                iteration=0,
            )
            await router.run(task, state)
            task_type = state["task"].task_type or "general"
            print(f'    "{q[:50]}..."')
            print(f"      -> {task_type}")


async def demo_oracle() -> None:
    """Demo the Oracle agent (forecasting)."""
    from pharos.agents.oracle import OracleAgent
    from pharos.config import get_settings
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.orchestration.task_models import Task, TaskType, WorkflowState
    from pharos.tools.ollama_client import OllamaClient
    from pharos.tools.pubmed_tools import PubMedClient

    _section("Oracle Agent (Forecasting)")
    settings = get_settings()
    settings.oracle_max_pubmed_queries = 10  # limit for demo
    ollama = OllamaClient(settings)
    pubmed = PubMedClient(settings)

    async with Neo4jManager(settings) as kg:
        oracle = OracleAgent(ollama, kg, settings, pubmed=pubmed)
        task = Task(
            query="Predict emerging therapeutic targets for Alzheimer's disease",
            task_type=TaskType.FORECAST,
        )
        state = WorkflowState(
            task=task,
            results=[],
            current_agent="oracle",
            kg_context="",
            iteration=0,
        )
        print("    Running forecast pipeline (this may take a minute)...")
        result = await oracle.run(task, state)
        print(f"    Confidence: {result.confidence:.0%}")
        entities = result.artifacts.get("entities_analysed", [])
        print(f"    Entities analysed: {', '.join(entities[:5])}")
        hypotheses = result.artifacts.get("hypotheses", [])
        print(f"    Hypotheses generated: {len(hypotheses)}")
        # Print first 200 chars of report
        preview = result.content[:200].replace("\n", " ")
        print(f"    Report preview: {preview}...")


async def demo_sentinel() -> None:
    """Demo the Sentinel agent (verification)."""
    from pharos.agents.sentinel import SentinelAgent
    from pharos.config import get_settings
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.orchestration.task_models import AgentResult, Task, TaskType, WorkflowState
    from pharos.tools.ollama_client import OllamaClient

    _section("Sentinel Agent (Verification)")
    settings = get_settings()
    ollama = OllamaClient(settings)

    async with Neo4jManager(settings) as kg:
        sentinel = SentinelAgent(ollama, kg, settings)

        # Create a fake result to verify
        fake_result = AgentResult(
            agent_name="Oracle",
            task_id="demo-1",
            content=(
                "KRAS mutations are found in approximately 25% of all human cancers. "
                "The G12C mutation is the most common KRAS variant in non-small cell "
                "lung cancer. Sotorasib (AMG 510) was the first KRAS G12C inhibitor "
                "approved by the FDA in 2021."
            ),
            confidence=0.8,
        )

        task = Task(query="verify", task_type=TaskType.VERIFY)
        state = WorkflowState(
            task=task,
            results=[fake_result],
            current_agent="sentinel",
            kg_context="",
            iteration=0,
        )

        print("    Verifying Oracle output...")
        result = await sentinel.run(task, state)
        print(f"    Verification score: {result.confidence:.0%}")
        verifications = result.artifacts.get("verifications", [])
        for v in verifications:
            print(f"    - {v['agent_name']}: {v['score']:.0%} ({len(v['checks'])} checks)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all demo steps."""
    _header("PHAROS Interactive Demo")
    print("  Platform for Hypothesis-driven Autonomous Research,")
    print("  Orchestration & Synthesis\n")

    # Pre-flight checks
    ollama_ok = await check_ollama()
    neo4j_ok = await check_neo4j()

    if not ollama_ok:
        print(f"\n{RED}Ollama is required. Start it with: docker-compose up -d{RESET}")
        sys.exit(1)

    if not neo4j_ok:
        print(f"\n{YELLOW}Neo4j not available — KG features will be limited.{RESET}")

    # Run demos
    try:
        await demo_router()
    except Exception as exc:
        _fail(f"Router demo: {exc}")

    try:
        await demo_oracle()
    except Exception as exc:
        _fail(f"Oracle demo: {exc}")

    if neo4j_ok:
        try:
            await demo_sentinel()
        except Exception as exc:
            _fail(f"Sentinel demo: {exc}")

    _header("Demo Complete")
    print("  Try the full UI:  chainlit run pharos/ui/chainlit_app.py")
    print("  Run all tests:    pytest tests/ -v\n")


if __name__ == "__main__":
    asyncio.run(main())
