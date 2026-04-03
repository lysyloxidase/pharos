# PHAROS

**Platform for Hypothesis-driven Autonomous Research, Orchestration & Synthesis**

> A multi-agent LLM platform for biomedical discovery, running **100% locally** via [Ollama](https://ollama.com) — zero cloud dependency, full data sovereignty.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-270%20passed-brightgreen.svg)]()
[![mypy](https://img.shields.io/badge/mypy-strict-green.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
                     User Query
                         |
                  +------v------+
                  |   Chainlit  |  Interactive chat + file upload
                  |     UI      |  PDF -> Scribe, PDB -> Architect
                  +------+------+
                         |
                  +------v------+
                  |   Router    |  Fast keyword match + LLM fallback
                  +------+------+
                         |
      +---------+--------+--------+-----------+----------+
      |         |        |        |           |          |
+-----v----++---v---++---v----++--v------++---v------++--v-------+
|  Oracle  || Scribe|| Carto- ||Alchemist|| Architect|| General  |
|Forecasts ||Reviews||grapher ||Molecules||  Proteins||  Q & A   |
|  Trends  ||& Cites||KG Build||  SMILES ||  ESM-2   || Fast LLM |
| Hypoths. ||PubMed ||  NER   ||  RDKit  || ProtGPT2 ||          |
+-----+----++---+---++---+----++--+------++---+------++--+-------+
      |         |        |        |           |          |
      +---------+--------+--------+-----------+      +---v-------+
                         |                           | Aggregate |
                  +------v------+                    |  & Output |
                  |  Sentinel   |                    +-----------+
                  | Verification|
                  +------+------+
                         |
             +-----------v-----------+
             |  score < 0.5?         |--yes--> retry specialist (max 1)
             +-----------+-----------+
                         | no
                  +------v------+
                  |  Aggregate  |  Merge results, update KG,
                  |  & Output   |  format Markdown report
                  +------+------+
                         |
                  +------v------+
                  |   Neo4j     |  Shared knowledge graph --
                  |     KG      |  persistent memory across agents
                  +-------------+
```

The **Knowledge Graph** acts as shared memory: every agent reads from and writes to the same Neo4j instance, enabling cross-agent reasoning and cumulative knowledge.

The **General** agent provides a fast path for simple questions (single LLM call, no PubMed/Sentinel overhead).

---

## Agents

| Agent | Role | Pipeline |
|-------|------|----------|
| **Router** | Classifies queries into 7 task types | Keyword match (instant) -> LLM fallback if ambiguous |
| **Oracle** | Biomedical forecasting & trend prediction | Entity extraction -> PubMed trends -> Embedding drift -> KG gap analysis -> Hypothesis generation -> Report |
| **Scribe** | Structured literature reviews with citations | Outline -> PubMed search -> Fetch abstracts -> Draft with citations -> Self-critique (3x) -> APA bibliography |
| **Cartographer** | Knowledge graph construction from literature | PubMed search -> Abstract fetch -> NER -> Relation extraction -> Deduplication -> Neo4j persistence |
| **Alchemist** | Small molecule design & optimization | LLM reasoning + RDKit validation *(stub)* |
| **Architect** | Protein engineering | ProtGPT2 / ProteinMPNN / ESM-2 -> Rank by perplexity -> Report |
| **Sentinel** | Verification & fact-checking (runs last) | Claim extraction -> KG/PubMed cross-ref -> Molecule/protein validation -> Citation check -> Hallucination detection |
| **General** | Fast answers to simple biomedical questions | Single LLM call, skips Sentinel |

---

## Models

All models run locally via Ollama. Configure in `.env` or environment variables.

| Model | Default | Role | RAM |
|-------|---------|------|-----|
| Router | `mistral` | Task classification | ~4 GB |
| Extractor | `mistral` | Entity extraction, drafting | ~4 GB |
| Verifier | `mistral` | Fact-checking, verification | ~4 GB |
| Reasoner | `mistral` | Hypothesis generation, reports | ~4 GB |
| Embeddings | `nomic-embed-text` | Semantic search, trend drift | ~0.5 GB |

> **Note:** The defaults above use `mistral` (7B) for all tasks, which works on consumer hardware. For better quality, use larger models — see [Model Recommendations](#model-recommendations).

### Model Recommendations

For higher quality output with more capable hardware:

| Role | Budget Model (8GB VRAM) | Quality Model (24GB+ VRAM) |
|------|------------------------|---------------------------|
| Router | `mistral` | `llama3.2:3b` |
| Extractor | `mistral` | `llama3.1:8b` |
| Verifier | `mistral` | `phi4:14b` |
| Reasoner | `mistral` | `llama3.3:70b` |
| Embeddings | `nomic-embed-text` | `nomic-embed-text` |

Additionally, the **Architect** agent uses these protein-specific models (loaded on-demand via HuggingFace):

| Model | Size | Role |
|-------|------|------|
| ESM-2 (650M) | ~4 GB | Embeddings, variant effects, perplexity |
| ProtGPT2 (738M) | ~4 GB | De-novo protein sequence generation |
| ESMFold | ~8 GB | 3D structure prediction |
| ProteinMPNN | ~150 MB | Inverse folding |

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** — [Download](https://ollama.com/download)
- **Docker** — [Download](https://www.docker.com/products/docker-desktop/) (for Neo4j)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/pharos.git
cd pharos
pip install -e ".[dev]"
```

### 2. Start Ollama & Pull Models

```bash
ollama serve  # skip if already running
ollama pull mistral
ollama pull nomic-embed-text
```

### 3. Start Neo4j

```bash
docker run -d --name neo4j \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/pharos_password \
  neo4j:5
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
PHAROS_NEO4J_URI=bolt://localhost:7687
PHAROS_NEO4J_USER=neo4j
PHAROS_NEO4J_PASSWORD=pharos_password
PHAROS_OLLAMA_HOST=http://localhost:11434
PHAROS_EMBEDDING_MODEL=nomic-embed-text
PHAROS_MODEL_ROUTER=mistral
PHAROS_MODEL_EXTRACTOR=mistral
PHAROS_MODEL_VERIFIER=mistral
PHAROS_MODEL_REASONER=mistral
```

### 5. Run

```bash
# Interactive demo (tests connections + sample queries)
python scripts/demo.py

# Full chat UI
chainlit run pharos/ui/chainlit_app.py
# Open http://localhost:8000
```

### 6. Test

```bash
pytest tests/ -v          # 270 tests
mypy pharos/ --strict     # 0 errors
ruff check .              # 0 warnings
```

---

## Usage Examples

**Simple question** (~10 seconds):
```
> What is lysyl oxidase?
```

**Forecasting** (~2-5 minutes):
```
> Predict emerging therapeutic targets for Alzheimer's disease
```

**Literature review** (~3-5 minutes):
```
> Write a review of CRISPR-Cas9 applications in oncology
```

**Knowledge graph** (~2-3 minutes):
```
> Build a knowledge graph of kinase inhibitors in breast cancer
```

**Protein design** (requires HuggingFace models):
```
> Design a thermostable variant of T4 lysozyme
```

---

## Hardware Requirements

| Tier | RAM | GPU VRAM | What You Get |
|------|-----|----------|--------------|
| **Minimum** | 16 GB | 8 GB | All agents with `mistral` (7B) |
| **Recommended** | 64 GB | 24 GB | Mixed models (8B/14B/32B) |
| **Full** | 128 GB | 48 GB+ | All agents with 70B reasoning |

- CPU-only works but is slower (especially protein structure prediction)
- Models are loaded lazily — only active models consume VRAM
- Ollama handles automatic model swapping when VRAM is limited

---

## Why Local?

| Benefit | Details |
|---------|---------|
| **Data Sovereignty** | Biomedical data stays on-premise. No sequences, patient data, or unpublished findings leave your network. |
| **Reproducibility** | Fixed model versions, deterministic inference with seed control. |
| **Cost** | Zero marginal cost per query after hardware investment. |
| **Latency** | No network round-trips for LLM calls. |
| **Compliance** | Meets HIPAA, GDPR, and institutional IRB requirements by design. |
| **Offline** | Works in air-gapped environments. Download models once, run anywhere. |

---

## Project Structure

```
pharos/
├── pharos/                    # Main package
│   ├── agents/                # 8 specialist agents
│   │   ├── base.py            #   Abstract base class
│   │   ├── router.py          #   Task classifier (keyword + LLM)
│   │   ├── oracle.py          #   Forecasting & trends
│   │   ├── scribe.py          #   Review writer
│   │   ├── cartographer.py    #   KG builder
│   │   ├── alchemist.py       #   Molecule designer (stub)
│   │   ├── architect.py       #   Protein engineer
│   │   ├── sentinel.py        #   Verification & fact-check
│   │   └── general.py         #   Fast Q&A agent
│   ├── forecasting/           # Trend detection & hypothesis gen
│   ├── graph/                 # Neo4j knowledge graph layer
│   ├── tools/                 # Ollama, PubMed, protein tools
│   ├── orchestration/         # LangGraph workflow engine
│   ├── ui/                    # Chainlit chat interface
│   └── config.py              # Pydantic Settings
├── tests/                     # 270+ pytest tests
├── scripts/                   # Demo & utility scripts
├── EXAMPLES.md                # 5 complete usage examples
├── CHANGELOG.md               # Development history
└── pyproject.toml             # Build config & tool settings
```

---

## Tech Stack

- **LLM Runtime**: [Ollama](https://ollama.com) (local inference)
- **Knowledge Graph**: [Neo4j](https://neo4j.com) (async driver)
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph)
- **UI**: [Chainlit](https://chainlit.io) (chat + file upload + streaming)
- **Data**: [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) (literature search)
- **Protein ML**: ESM-2, ProtGPT2, ProteinMPNN, ESMFold (HuggingFace)
- **Chemistry**: [RDKit](https://www.rdkit.org/) (molecular validation)
- **Config**: [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) (env-based)
- **Quality**: ruff, mypy --strict, pytest (270+ tests)

---

## Development

```bash
ruff check pharos/ tests/       # Lint
ruff format pharos/ tests/      # Format
mypy pharos/ --strict           # Type check (0 errors)
pytest tests/ -v --tb=short     # Test (270 passing)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your changes
4. Ensure all checks pass: `ruff check . && mypy pharos/ --strict && pytest tests/ -v`
5. Submit a pull request

Code style: Google docstrings, type hints everywhere, async-first, Pydantic v2 models.

---

## License

[MIT](LICENSE)
