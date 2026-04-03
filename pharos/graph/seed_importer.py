"""Import seed data from PrimeKG / Hetionet into the PHAROS knowledge graph.

TODO (Phase 2):
    - Download and parse PrimeKG CSV files
    - Map PrimeKG node/edge types to PHAROS schema
    - Batch import with progress tracking
    - Hetionet JSON import as alternative source
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pharos.graph.neo4j_manager import Neo4jManager


async def import_primekg(kg: Neo4jManager, data_dir: str) -> dict[str, int]:
    """Import PrimeKG dataset into the knowledge graph.

    Args:
        kg: Neo4j manager instance.
        data_dir: Path to directory containing PrimeKG CSV files.

    Returns:
        Dict with counts of imported nodes and relations.

    Raises:
        NotImplementedError: PrimeKG import is not yet implemented.
    """
    raise NotImplementedError(
        "PrimeKG import will be implemented in Phase 2. "
        "Expected files: nodes.csv, edges.csv in the data directory."
    )


async def import_hetionet(kg: Neo4jManager, json_path: str) -> dict[str, int]:
    """Import Hetionet dataset into the knowledge graph.

    Args:
        kg: Neo4j manager instance.
        json_path: Path to Hetionet JSON file.

    Returns:
        Dict with counts of imported nodes and relations.

    Raises:
        NotImplementedError: Hetionet import is not yet implemented.
    """
    raise NotImplementedError(
        "Hetionet import will be implemented in Phase 2. Expected input: hetionet-v1.0.json"
    )
