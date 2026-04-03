#!/usr/bin/env bash
# Start Neo4j Community Edition via Docker.
#
# Usage: bash scripts/setup_neo4j.sh
#
# Requires: Docker installed and running.

set -euo pipefail

NEO4J_PASSWORD="${NEO4J_PASSWORD:-pharos-dev}"
NEO4J_VERSION="5-community"

echo "=== PHAROS Neo4j Setup ==="
echo "Starting Neo4j ${NEO4J_VERSION} ..."

docker run -d \
    --name pharos-neo4j \
    --restart unless-stopped \
    -p 7474:7474 \
    -p 7687:7687 \
    -e NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" \
    -e NEO4J_PLUGINS='["apoc"]' \
    -v pharos-neo4j-data:/data \
    -v pharos-neo4j-logs:/logs \
    "neo4j:${NEO4J_VERSION}"

echo ""
echo "Neo4j is starting..."
echo "  Browser: http://localhost:7474"
echo "  Bolt:    bolt://localhost:7687"
echo "  User:    neo4j"
echo "  Pass:    ${NEO4J_PASSWORD}"
echo ""
echo "Wait ~30s for Neo4j to become ready."
