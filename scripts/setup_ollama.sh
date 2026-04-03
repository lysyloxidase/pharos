#!/usr/bin/env bash
# Pull all Ollama models required by PHAROS.
#
# Usage: bash scripts/setup_ollama.sh
#
# Requires: ollama CLI installed and running (ollama serve)

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

MODELS=(
    "llama3.2:3b"       # Router — fast classification (~2 GB)
    "llama3.1:8b"       # Extractor, Oracle, Scribe, Cartographer (~5 GB)
    "phi4:14b"          # Sentinel — verification (~9 GB)
    "all-minilm:l6-v2"  # Embeddings (~0.5 GB)
)

# Optional large models (uncomment if you have >=48 GB VRAM)
# MODELS+=("llama3.3:70b")           # Reasoner — Architect (~42 GB)
# MODELS+=("qwen2.5-coder:32b")     # Coder — Alchemist (~20 GB)

echo "=== PHAROS Ollama Model Setup ==="
echo "Host: ${OLLAMA_HOST}"
echo ""

for model in "${MODELS[@]}"; do
    echo ">>> Pulling ${model} ..."
    ollama pull "${model}"
    echo "    Done."
    echo ""
done

echo "=== All models pulled successfully ==="
ollama list
