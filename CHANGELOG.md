# Changelog

All notable changes to the PHAROS project.

## [0.1.0] - 2026-04-03

### Phase 1 — Foundation

- Project scaffolding: 49 files, package structure, CI pipeline
- **Router agent** (`agents/router.py`) — LLM task classification with keyword fallback, sub-task decomposition
- **Neo4j manager** (`graph/neo4j_manager.py`) — async CRUD, schema setup, fulltext search, node deduplication
- **Ollama client** (`tools/ollama_client.py`) — async generate, stream, chat, embed with retry and timeout
- **LangGraph workflow** (`orchestration/graph_workflow.py`) — StateGraph with conditional routing
- **Task models** (`orchestration/task_models.py`) — Pydantic v2 models for Task, AgentResult, WorkflowState
- **Config** (`config.py`) — pydantic-settings with `PHAROS_` env prefix
- Docker Compose for Neo4j + Ollama, setup scripts, CI workflow

### Phase 2 — Data Layer

- **PubMed client** (`tools/pubmed_tools.py`) — E-utilities search/fetch with rate limiting (3/s default, 10/s with API key), retry on 429/503, XML parsing
- **Entity extractor** (`graph/entity_extractor.py`) — LLM-based NER for 7 biomedical entity types, relation extraction with confidence scoring, JSON extraction with code-fence handling
- **Cartographer agent** (`agents/cartographer.py`) — full pipeline: PubMed search, abstract fetch, entity/relation extraction, deduplication, KG persistence

### Phase 3 — Scribe Agent

- **Scribe agent** (`agents/scribe.py`) — 6-step review writing pipeline: outline generation, PubMed search queries, abstract retrieval, section drafting with inline citations, self-critique loop (max 3 iterations, score threshold 7/10), abstract generation, Markdown assembly with APA bibliography
- Citation management with PMID-backed references and deduplication

### Phase 4 — Oracle Agent & Forecasting

- **Trend detector** (`forecasting/trend_detector.py`) — publication velocity (linear regression), acceleration (split-half), semantic drift (cosine distance of yearly embedding centroids), entity convergence signals
- **Hypothesis generator** (`forecasting/hypothesis_gen.py`) — KG gap analysis (diseases without drugs, high-connectivity targets), convergence signals, LLM synthesis
- **Oracle agent** (`agents/oracle.py`) — full pipeline: entity extraction, trend computation, hypothesis generation, narrative report
- Shelve-based embedding cache, configurable PubMed query limits

### Phase 5 — Architect Agent & Protein Tools

- **Protein toolkit** (`tools/protein_tools.py`) — ESM-2 embeddings and variant effects (masked marginal scoring), ProtGPT2 de-novo generation, ProteinMPNN inverse folding (subprocess), ESMFold structure prediction, pseudo-perplexity scoring, sequence identity and validation
- **Architect agent** (`agents/architect.py`) — 7-step pipeline: parse design brief, KG context query, LLM strategy decision (de novo / redesign / mutagenesis), strategy execution, ESM-2 perplexity ranking, report generation, KG persistence
- Lazy model loading for ESM-2 (~4 GB) and ProtGPT2 (~4 GB)
- Graceful degradation when ProteinMPNN is not installed

### Phase 6 — Sentinel, Integration & Polish

- **Sentinel agent** (`agents/sentinel.py`) — 5-check verification: claim extraction + KG/PubMed fact-checking, RDKit molecule validation, protein sequence validation, PMID citation verification, LLM hallucination detection
- **Workflow update** — Sentinel retry logic (re-runs specialist if score < 0.5, max 1 retry), Markdown output aggregation
- **Chainlit UI** (`ui/chainlit_app.py`) — full implementation: chat interface, KG stats sidebar, file upload (PDF/PDB), streaming partial results, Markdown rendering
- **Integration tests** — full pipeline simulations for forecast, review, protein design, and retry flows
- README rewrite, EXAMPLES.md, demo script, CHANGELOG
- `__all__` exports in all `__init__.py` files
- 270+ tests passing, ruff clean, mypy strict compatible
