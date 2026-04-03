# PHAROS Usage Examples

Five complete examples demonstrating each major agent capability.

---

## 1. Predict Emerging Targets for Alzheimer's Disease

**Agent**: Oracle (Forecasting)

```python
from pharos.config import get_settings
from pharos.tools.ollama_client import OllamaClient
from pharos.tools.pubmed_tools import PubMedClient
from pharos.graph.neo4j_manager import Neo4jManager
from pharos.agents.oracle import OracleAgent
from pharos.orchestration.task_models import Task, WorkflowState, TaskType

settings = get_settings()
ollama = OllamaClient(settings)
pubmed = PubMedClient(settings)

async with Neo4jManager(settings) as kg:
    oracle = OracleAgent(ollama, kg, settings, pubmed=pubmed)
    task = Task(query="Predict emerging therapeutic targets for Alzheimer's disease")
    state = WorkflowState(
        task=task, results=[], current_agent="oracle",
        kg_context="", iteration=0,
    )
    result = await oracle.run(task, state)
    print(result.content)
```

**Example output:**

```markdown
# Forecasting Report

## Executive Summary
Analysis of publication trends from 2015-2026 reveals accelerating interest
in neuroinflammation-related targets for Alzheimer's disease, with TREM2
and CD33 showing the strongest velocity increases.

## Trend Data
- **TREM2**: velocity=12.3 pub/yr, acceleration=3.1, semantic_drift=0.187
  Emerging associations: microglia (0.28), neuroinflammation (0.22)
- **CD33**: velocity=8.7 pub/yr, acceleration=2.4, semantic_drift=0.143
- **APOE4**: velocity=15.1 pub/yr, acceleration=-1.2, semantic_drift=0.092

## Generated Hypotheses
1. [75% confidence, 3-5 years] TREM2 agonists may slow neurodegeneration
   by modulating microglial phagocytosis of amyloid plaques.
2. [68% confidence, 1-2 years] CD33 inhibition combined with anti-amyloid
   therapy may show synergistic effects in early-stage AD.
3. [62% confidence, 3-5 years] Gut microbiome modulation via SCFA
   supplementation may reduce neuroinflammation through the gut-brain axis.
```

---

## 2. Write a Narrative Review on CRISPR Delivery Methods

**Agent**: Scribe (Review Writer)

```python
from pharos.agents.scribe import ScribeAgent

async with Neo4jManager(settings) as kg:
    scribe = ScribeAgent(ollama, kg, settings, pubmed=pubmed)
    task = Task(query="Write a narrative review on CRISPR-Cas9 delivery methods for in vivo gene therapy")
    state = WorkflowState(
        task=task, results=[], current_agent="scribe",
        kg_context="", iteration=0,
    )
    result = await scribe.run(task, state)
    print(result.content)
```

**Example output:**

```markdown
# CRISPR-Cas9 Delivery Methods for In Vivo Gene Therapy: A Narrative Review

## Abstract
This review examines current delivery strategies for CRISPR-Cas9 systems
in vivo, covering viral vectors (AAV, lentivirus), lipid nanoparticles,
and emerging physical methods. We discuss tissue-specific targeting,
immunogenicity concerns, and clinical translation challenges...

## 1. Introduction
The advent of CRISPR-Cas9 genome editing has transformed the landscape of
gene therapy [Doudna & Charpentier, 2014]. However, efficient delivery of
the Cas9 ribonucleoprotein to target tissues remains the primary bottleneck
for clinical translation [Yin et al., 2017]...

## 2. Viral Delivery Vectors
Adeno-associated virus (AAV) vectors remain the most clinically advanced
delivery platform for CRISPR components. AAV serotypes exhibit distinct
tissue tropisms: AAV9 crosses the blood-brain barrier [Mendell et al., 2017],
while AAV8 targets hepatocytes [Ran et al., 2015]...

## 3. Non-Viral Delivery Systems
Lipid nanoparticles (LNPs) have emerged as a versatile alternative,
particularly following the success of mRNA vaccines [Gillmore et al., 2021].
LNP-mediated delivery of Cas9 mRNA and sgRNA achieved >70% editing
efficiency in primate hepatocytes...

## References
- Doudna, J. A. & Charpentier, E. (2014). The new frontier of genome
  engineering with CRISPR-Cas9. *Science*, 346(6213). PMID: 25430774
- Gillmore, J. D. et al. (2021). CRISPR-Cas9 In Vivo Gene Editing for
  Transthyretin Amyloidosis. *NEJM*, 385(6). PMID: 34215024
```

---

## 3. Build a Knowledge Graph of Kinase Inhibitors

**Agent**: Cartographer (KG Builder)

```python
from pharos.agents.cartographer import CartographerAgent
from pharos.graph.entity_extractor import BioEntityExtractor

async with Neo4jManager(settings) as kg:
    extractor = BioEntityExtractor(ollama, settings)
    cartographer = CartographerAgent(
        ollama, kg, settings, pubmed=pubmed, extractor=extractor
    )
    task = Task(query="Build a knowledge graph of kinase inhibitors and their targets")
    state = WorkflowState(
        task=task, results=[], current_agent="cartographer",
        kg_context="", iteration=0,
    )
    result = await cartographer.run(task, state)
    print(result.content)
    print(f"Entities: {result.artifacts.get('entities_added', 0)}")
    print(f"Relations: {result.artifacts.get('relations_added', 0)}")
```

**Example output:**

```markdown
# Knowledge Graph Construction Report

## Summary
Processed 45 PubMed abstracts on kinase inhibitors. Extracted and
persisted 78 entities and 124 relations to the knowledge graph.

## Key Entities Added
- **Drugs**: imatinib, dasatinib, sunitinib, erlotinib, vemurafenib, ...
- **Genes/Proteins**: BCR-ABL, EGFR, BRAF, VEGFR, KIT, PDGFR, ...
- **Diseases**: CML, NSCLC, melanoma, GIST, renal cell carcinoma, ...
- **Pathways**: MAPK/ERK, PI3K/AKT, JAK/STAT, ...

## Sample Relations
- imatinib —[INHIBITS]→ BCR-ABL (confidence: 0.95)
- imatinib —[TREATS]→ CML (confidence: 0.92)
- vemurafenib —[INHIBITS]→ BRAF V600E (confidence: 0.94)
- erlotinib —[INHIBITS]→ EGFR (confidence: 0.91)
- BRAF V600E —[CAUSES]→ melanoma (confidence: 0.87)

## KG Statistics
- Total nodes: 312 (+78)
- Total relations: 587 (+124)
```

---

## 4. Design a Drug-like Molecule Targeting EGFR

**Agent**: Alchemist (Molecular Designer)

> Note: The Alchemist agent is a stub in v0.1. This example shows the
> planned interface and expected output format.

```python
from pharos.agents.alchemist import AlchemistAgent

async with Neo4jManager(settings) as kg:
    alchemist = AlchemistAgent(ollama, kg, settings)
    task = Task(
        query="Design a selective EGFR inhibitor with good oral bioavailability, "
              "MW < 500, logP 1-3, no PAINS alerts"
    )
    state = WorkflowState(
        task=task, results=[], current_agent="alchemist",
        kg_context="", iteration=0,
    )
    result = await alchemist.run(task, state)
```

**Planned output format:**

```markdown
# Molecular Design Report

## Design Brief
Target: EGFR (Epidermal Growth Factor Receptor)
Constraints: MW < 500, logP 1-3, no PAINS alerts, oral bioavailability

## Top Candidates

| # | SMILES | MW | logP | TPSA | Lipinski | Synth. Access. |
|---|--------|-----|------|------|----------|----------------|
| 1 | c1cc(NC(=O)...)... | 432.5 | 2.1 | 89.4 | Pass | 3.2 |
| 2 | c1cc(F)c(NC...)... | 467.3 | 2.8 | 76.2 | Pass | 3.8 |
| 3 | c1cnc(NC(=O)...).. | 445.1 | 1.9 | 95.1 | Pass | 3.5 |

## Structure-Activity Analysis
The quinazoline scaffold present in candidates 1 and 3 mimics the
ATP-binding site geometry of EGFR, consistent with known inhibitors
erlotinib and gefitinib...
```

---

## 5. Design a Thermostable Variant of T4 Lysozyme

**Agent**: Architect (Protein Engineer)

```python
from pharos.agents.architect import ArchitectAgent

async with Neo4jManager(settings) as kg:
    architect = ArchitectAgent(ollama, kg, settings)

    # Mutagenesis strategy — provide the WT sequence and mutations
    t4l_sequence = (
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVIT"
        "KDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSL"
        "RMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
    )

    task = Task(
        query=f"Suggest stabilising mutations for T4 lysozyme. "
              f"Sequence: {t4l_sequence}. "
              f"Mutations to test: L99A, T152V, A98L, I3C"
    )
    state = WorkflowState(
        task=task, results=[], current_agent="architect",
        kg_context="", iteration=0,
    )
    result = await architect.run(task, state)
    print(result.content)

    # Check variant effects
    for ve in result.artifacts.get("variant_effects", []):
        print(f"  {ve['mutation']}: {ve['predicted_effect']} "
              f"(delta_ll={ve['delta_log_likelihood']:.3f})")
```

**Example output:**

```markdown
# Protein Design Report

## Design Brief
Design stabilising mutations for T4 lysozyme (164 residues).

## Strategy
Mutagenesis — directed evolution using ESM-2 masked marginal scoring
to evaluate the effect of each proposed mutation on protein stability.

## Variant Effects
| Mutation | Effect | Delta Log-Likelihood |
|----------|--------|---------------------|
| A98L     | stabilizing | -1.23 |
| T152V    | stabilizing | -0.87 |
| L99A     | destabilizing | +0.65 |
| I3C      | neutral | -0.12 |

## Candidates (ranked by perplexity)
1. WT + A98L + T152V (ppl=4.8, len=164)
2. WT + A98L (ppl=5.1, len=164)
3. WT (ppl=5.3, len=164)

## Recommendations
- **A98L** and **T152V** are predicted to be stabilising — prioritise
  these for experimental validation.
- **L99A** creates a cavity in the hydrophobic core; expected to
  decrease thermal stability (consistent with known literature).
- Consider combining A98L + T152V for additive stabilisation.
```

---

## Running These Examples

All examples require:
1. Ollama running with the required models pulled
2. Neo4j running (via `docker-compose up -d`)
3. PHAROS installed (`pip install -e ".[dev]"`)

For a quick interactive test of all agents, run:

```bash
python scripts/demo.py
```
