#!/usr/bin/env python3
"""Import seed data from PrimeKG into the PHAROS knowledge graph.

Usage::

    python scripts/seed_graph.py --data-dir ./data/primekg

Requires the PrimeKG dataset to be downloaded first.
See: https://github.com/mims-harvard/PrimeKG
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def main(data_dir: str) -> None:
    """Run the PrimeKG seed import.

    Args:
        data_dir: Path to directory containing PrimeKG CSV files.
    """
    from pharos.config import get_settings
    from pharos.graph.neo4j_manager import Neo4jManager
    from pharos.graph.seed_importer import import_primekg

    settings = get_settings()
    async with Neo4jManager(settings) as kg:
        await kg.setup_schema()
        counts = await import_primekg(kg, data_dir)
        print(f"Imported: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed PHAROS KG from PrimeKG")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to PrimeKG data directory",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.data_dir))
    except NotImplementedError as e:
        print(f"Not yet implemented: {e}", file=sys.stderr)
        sys.exit(1)
