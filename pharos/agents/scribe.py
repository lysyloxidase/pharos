"""Agent 2 — Scribe: narrative review writer.

Generates structured literature reviews with PubMed citations by:
1. Outlining sections via LLM
2. Searching PubMed per section
3. Drafting each section with inline citations
4. Self-critiquing and rewriting weak sections
5. Assembling the final Markdown document with bibliography
6. Enriching the knowledge graph with extracted entities
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from pharos.agents.base import BaseAgent
from pharos.graph.entity_extractor import BioEntityExtractor, _extract_json
from pharos.orchestration.prompts import PROMPTS
from pharos.orchestration.task_models import AgentResult, Task, WorkflowState
from pharos.tools.pubmed_tools import PubMedArticle, PubMedClient

logger = logging.getLogger(__name__)

_MAX_CRITIQUE_ITERATIONS = 3
_ARTICLES_PER_SECTION = 10
_MIN_ACCEPTABLE_SCORE = 7


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


class ReviewSection(BaseModel):
    """A single section in the review outline.

    Attributes:
        heading: Section heading text.
        key_questions: Questions the section should answer.
        search_queries: PubMed queries to find relevant literature.
    """

    heading: str
    key_questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class ReviewOutline(BaseModel):
    """The structural outline for a narrative review.

    Attributes:
        title: Review title.
        sections: Ordered list of section outlines.
    """

    title: str
    sections: list[ReviewSection]


class Reference(BaseModel):
    """A bibliographic reference.

    Attributes:
        key: Short citation key (e.g. "Smith2024").
        pmid: PubMed identifier.
        authors: Author names.
        title: Article title.
        journal: Journal name.
        year: Publication year.
        doi: DOI if available.
    """

    key: str
    pmid: str
    authors: list[str] = Field(default_factory=list)
    title: str = ""
    journal: str = ""
    year: int = 0
    doi: str | None = None


class DraftSection(BaseModel):
    """A drafted section of the review.

    Attributes:
        heading: Section heading.
        body: Markdown body text with inline citations.
        score: Quality score from self-critique (1-10).
    """

    heading: str
    body: str
    score: int = 0


class ReviewDraft(BaseModel):
    """The complete review document.

    Attributes:
        title: Review title.
        abstract: Generated abstract.
        sections: Drafted sections.
        references: Bibliography entries.
    """

    title: str
    abstract: str = ""
    sections: list[DraftSection] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)


# ------------------------------------------------------------------
# Scribe Agent
# ------------------------------------------------------------------


class ScribeAgent(BaseAgent):
    """Narrative review writing agent.

    Orchestrates a multi-step pipeline that searches PubMed, drafts
    sections with inline citations, self-critiques, and assembles a
    complete Markdown review with bibliography.

    The ``pubmed`` and ``extractor`` dependencies can be injected for testing.
    """

    def __init__(
        self,
        ollama: Any,
        kg: Any,
        config: Any,
        *,
        pubmed: PubMedClient | None = None,
        extractor: BioEntityExtractor | None = None,
    ) -> None:
        super().__init__(ollama, kg, config)
        self.pubmed = pubmed or PubMedClient(config)
        self.extractor = extractor or BioEntityExtractor(ollama, config)

    async def run(self, task: Task, state: WorkflowState) -> AgentResult:
        """Execute the full review-writing pipeline.

        Args:
            task: Task whose ``query`` describes the review topic.
            state: Current workflow state.

        Returns:
            AgentResult with Markdown review and structured artifacts.
        """
        task_id = str(uuid.uuid4())
        topic = task.query

        # 1. Generate outline
        outline = await self._generate_outline(topic)
        if not outline.sections:
            return AgentResult(
                agent_name="Scribe",
                task_id=task_id,
                content="Failed to generate review outline.",
                confidence=0.1,
            )

        # 2. Generate search queries per section
        outline = await self._generate_search_queries(outline)

        # 3. Search PubMed & collect references per section
        section_articles: list[list[PubMedArticle]] = []
        all_articles: dict[str, PubMedArticle] = {}  # pmid -> article
        for section in outline.sections:
            articles = await self._search_for_section(section)
            section_articles.append(articles)
            for a in articles:
                all_articles[a.pmid] = a

        # Build reference registry
        references = _build_references(list(all_articles.values()))
        ref_by_pmid = {r.pmid: r for r in references}

        # 4. Retrieve KG context
        kg_context = await self._get_kg_context(topic)

        # 5. Draft each section with critique loop
        draft_sections: list[DraftSection] = []
        for section, articles in zip(outline.sections, section_articles, strict=True):
            section_refs = [ref_by_pmid[a.pmid] for a in articles if a.pmid in ref_by_pmid]
            drafted = await self._draft_section(section, articles, section_refs, kg_context)
            draft_sections.append(drafted)

        # 6. Generate abstract
        full_body = "\n\n".join(f"## {s.heading}\n\n{s.body}" for s in draft_sections)
        abstract = await self._generate_abstract(full_body)

        # 7. Assemble final document
        draft = ReviewDraft(
            title=outline.title,
            abstract=abstract,
            sections=draft_sections,
            references=references,
        )
        markdown = _assemble_markdown(draft)

        # 8. KG enrichment
        kg_updates = await self._enrich_kg(full_body)

        return AgentResult(
            agent_name="Scribe",
            task_id=task_id,
            content=markdown,
            artifacts={
                "title": draft.title,
                "sections": [s.heading for s in draft.sections],
                "reference_count": len(draft.references),
                "section_scores": [s.score for s in draft.sections],
            },
            confidence=min(
                0.95,
                sum(s.score for s in draft_sections) / max(len(draft_sections), 1) / 10,
            ),
            kg_updates=kg_updates,
        )

    # ------------------------------------------------------------------
    # Step 1: Outline generation
    # ------------------------------------------------------------------

    async def _generate_outline(self, topic: str) -> ReviewOutline:
        """Generate a structured outline for the review.

        Args:
            topic: Review topic string.

        Returns:
            A ReviewOutline with title and sections.
        """
        raw = await self.ollama.generate(
            model=self.config.model_reasoner,
            prompt=f"Create an outline for a narrative review on: {topic}",
            system=PROMPTS["scribe_outline"],
            format="json",
        )
        return _parse_outline(raw)

    # ------------------------------------------------------------------
    # Step 2: Search query generation
    # ------------------------------------------------------------------

    async def _generate_search_queries(self, outline: ReviewOutline) -> ReviewOutline:
        """Generate PubMed search queries for each section.

        Args:
            outline: Outline with sections (queries will be filled in).

        Returns:
            Updated outline with search_queries populated.
        """
        for section in outline.sections:
            prompt = (
                f"Section heading: {section.heading}\n"
                f"Key questions: {json.dumps(section.key_questions)}\n"
                f"Generate PubMed search queries."
            )
            raw = await self.ollama.generate(
                model=self.config.model_extractor,
                prompt=prompt,
                system=PROMPTS["scribe_search"],
                format="json",
            )
            queries = _parse_search_queries(raw)
            if queries:
                section.search_queries = queries
            elif not section.search_queries:
                # Fallback: use heading as query
                section.search_queries = [section.heading]
        return outline

    # ------------------------------------------------------------------
    # Step 3: PubMed search per section
    # ------------------------------------------------------------------

    async def _search_for_section(self, section: ReviewSection) -> list[PubMedArticle]:
        """Search PubMed for articles relevant to a section.

        Args:
            section: Section outline with search queries.

        Returns:
            Deduplicated list of PubMedArticle objects.
        """
        all_pmids: list[str] = []
        for query in section.search_queries:
            pmids = await self.pubmed.search(query, max_results=_ARTICLES_PER_SECTION)
            all_pmids.extend(pmids)

        # Deduplicate, preserve order, limit
        seen: set[str] = set()
        unique_pmids: list[str] = []
        for pmid in all_pmids:
            if pmid not in seen:
                seen.add(pmid)
                unique_pmids.append(pmid)
        unique_pmids = unique_pmids[:_ARTICLES_PER_SECTION]

        if not unique_pmids:
            return []

        return await self.pubmed.fetch_abstracts(unique_pmids)

    # ------------------------------------------------------------------
    # Step 4: KG context retrieval
    # ------------------------------------------------------------------

    async def _get_kg_context(self, topic: str) -> str:
        """Query the knowledge graph for context relevant to the topic.

        Args:
            topic: Review topic string.

        Returns:
            Formatted string of KG facts, or empty string.
        """
        try:
            results = await self.kg.search_nodes(topic, limit=20)
            if not results:
                return ""
            lines: list[str] = []
            for r in results:
                node = r.get("n", r)
                if isinstance(node, dict):
                    name = node.get("name", "")
                    labels = node.get("_labels", [])
                    label_str = ", ".join(labels) if isinstance(labels, list) else str(labels)
                    lines.append(f"- {name} ({label_str})")
            return "Known entities from knowledge graph:\n" + "\n".join(lines)
        except Exception:
            logger.warning("KG context retrieval failed, proceeding without")
            return ""

    # ------------------------------------------------------------------
    # Step 5: Section drafting with self-critique
    # ------------------------------------------------------------------

    async def _draft_section(
        self,
        section: ReviewSection,
        articles: list[PubMedArticle],
        references: list[Reference],
        kg_context: str,
    ) -> DraftSection:
        """Draft a section and refine it through self-critique.

        Args:
            section: Section outline.
            articles: Articles retrieved for this section.
            references: Reference objects for citation keys.
            kg_context: KG context string.

        Returns:
            Finalized DraftSection with quality score.
        """
        # Build context for the LLM
        abstracts_context = _format_abstracts_context(articles, references)
        ref_keys = [f"[{r.key}]" for r in references]

        prompt = (
            f"Section heading: {section.heading}\n"
            f"Key questions: {', '.join(section.key_questions)}\n\n"
            f"Available citation keys: {', '.join(ref_keys)}\n\n"
            f"--- ABSTRACTS ---\n{abstracts_context}\n\n"
        )
        if kg_context:
            prompt += f"--- KNOWLEDGE GRAPH CONTEXT ---\n{kg_context}\n\n"
        prompt += "Write the section body with inline citations."

        body = await self.ollama.generate(
            model=self.config.model_reasoner,
            prompt=prompt,
            system=PROMPTS["scribe_draft"],
        )

        # Self-critique loop
        score = 0
        for iteration in range(_MAX_CRITIQUE_ITERATIONS):
            score, issues = await self._critique_section(body, abstracts_context)
            if score >= _MIN_ACCEPTABLE_SCORE:
                break
            # Don't rewrite after the last allowed iteration
            if iteration >= _MAX_CRITIQUE_ITERATIONS - 1:
                break
            # Rewrite
            rewrite_prompt = (
                f"Rewrite this section to fix these issues:\n"
                f"{json.dumps(issues)}\n\n"
                f"Original section:\n{body}\n\n"
                f"Available citation keys: {', '.join(ref_keys)}\n\n"
                f"--- ABSTRACTS ---\n{abstracts_context}"
            )
            body = await self.ollama.generate(
                model=self.config.model_reasoner,
                prompt=rewrite_prompt,
                system=PROMPTS["scribe_draft"],
            )

        return DraftSection(heading=section.heading, body=body.strip(), score=score)

    async def _critique_section(self, section_body: str, sources: str) -> tuple[int, list[str]]:
        """Critique a section draft for quality.

        Args:
            section_body: The drafted section text.
            sources: The source abstracts provided to the writer.

        Returns:
            Tuple of (score 1-10, list of issue strings).
        """
        prompt = f"--- SECTION TEXT ---\n{section_body}\n\n--- SOURCES ---\n{sources}"
        raw = await self.ollama.generate(
            model=self.config.model_verifier,
            prompt=prompt,
            system=PROMPTS["scribe_critique"],
            format="json",
        )
        return _parse_critique(raw)

    # ------------------------------------------------------------------
    # Step 6: Abstract generation
    # ------------------------------------------------------------------

    async def _generate_abstract(self, full_body: str) -> str:
        """Generate an abstract from the complete review body.

        Args:
            full_body: Concatenated Markdown body of all sections.

        Returns:
            Abstract text.
        """
        # Truncate if very long to fit context window
        truncated = full_body[:6000] if len(full_body) > 6000 else full_body
        return await self.ollama.generate(
            model=self.config.model_reasoner,
            prompt=f"Review article:\n\n{truncated}",
            system=PROMPTS["scribe_abstract"],
        )

    # ------------------------------------------------------------------
    # Step 7: KG enrichment
    # ------------------------------------------------------------------

    async def _enrich_kg(self, text: str) -> list[dict[str, Any]]:
        """Extract entities and relations from the review and prepare KG updates.

        Args:
            text: Full review body text.

        Returns:
            List of KG triple dicts.
        """
        try:
            entities = await self.extractor.extract_entities(text)
            relations = await self.extractor.extract_relations(text, entities)
        except Exception:
            logger.warning("KG enrichment failed")
            return []

        kg_updates: list[dict[str, Any]] = []
        for rel in relations:
            kg_updates.append(
                {
                    "source": {"type": "Entity", "properties": {"name": rel.source}},
                    "relation": rel.relation_type,
                    "target": {"type": "Entity", "properties": {"name": rel.target}},
                    "properties": {"confidence": str(rel.confidence)},
                }
            )
        return kg_updates


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _parse_outline(raw: str) -> ReviewOutline:
    """Parse LLM output into a ReviewOutline.

    Args:
        raw: Raw LLM JSON response.

    Returns:
        Parsed ReviewOutline, or an empty one on failure.
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return ReviewOutline(title="Untitled Review", sections=[])

    title = str(data.get("title", "Untitled Review"))
    raw_sections = data.get("sections", [])
    if not isinstance(raw_sections, list):
        return ReviewOutline(title=title, sections=[])

    sections: list[ReviewSection] = []
    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading", ""))
        if not heading:
            continue
        key_questions = [str(q) for q in item.get("key_questions", []) if isinstance(q, str)]
        sections.append(ReviewSection(heading=heading, key_questions=key_questions))

    return ReviewOutline(title=title, sections=sections)


def _parse_search_queries(raw: str) -> list[str]:
    """Parse LLM output into a list of PubMed query strings.

    Args:
        raw: Raw LLM JSON response.

    Returns:
        List of query strings.
    """
    data = _extract_json(raw)
    if isinstance(data, list):
        return [str(q) for q in data if isinstance(q, str)]
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return [str(q) for q in v if isinstance(q, str)]
    return []


def _parse_critique(raw: str) -> tuple[int, list[str]]:
    """Parse LLM critique JSON into score and issues.

    Args:
        raw: Raw LLM JSON response.

    Returns:
        Tuple of (score 1-10, list of issue strings).
        Defaults to (5, ["Failed to parse critique"]) on error.
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return 5, ["Failed to parse critique"]

    try:
        score = int(data.get("score", 5))
    except (ValueError, TypeError):
        score = 5
    score = max(1, min(10, score))

    issues_raw = data.get("issues", [])
    issues = [str(i) for i in issues_raw] if isinstance(issues_raw, list) else []

    return score, issues


def _build_references(articles: list[PubMedArticle]) -> list[Reference]:
    """Convert PubMedArticles to Reference objects with citation keys.

    Citation keys are formatted as "LastnameYear" (e.g. "Smith2024").
    Duplicates get a lowercase suffix (Smith2024a, Smith2024b).

    Args:
        articles: List of PubMed articles.

    Returns:
        List of Reference objects with unique citation keys.
    """
    refs: list[Reference] = []
    key_counts: dict[str, int] = {}

    for article in articles:
        # Build base key from first author + year
        last_name = article.authors[0].split()[0] if article.authors else "Unknown"
        base_key = f"{last_name}{article.year}"

        # Ensure uniqueness
        if base_key in key_counts:
            key_counts[base_key] += 1
            key = f"{base_key}{chr(96 + key_counts[base_key])}"  # a, b, c...
        else:
            key_counts[base_key] = 0
            key = base_key

        refs.append(
            Reference(
                key=key,
                pmid=article.pmid,
                authors=article.authors,
                title=article.title,
                journal=article.journal,
                year=article.year,
                doi=article.doi,
            )
        )
    return refs


def _format_abstracts_context(articles: list[PubMedArticle], references: list[Reference]) -> str:
    """Format articles and their citation keys into context for the LLM.

    Args:
        articles: Articles for this section.
        references: Corresponding Reference objects.

    Returns:
        Formatted context string.
    """
    ref_by_pmid = {r.pmid: r for r in references}
    parts: list[str] = []
    for article in articles:
        ref = ref_by_pmid.get(article.pmid)
        key = ref.key if ref else article.pmid
        authors_str = ", ".join(article.authors[:3])
        if len(article.authors) > 3:
            authors_str += " et al."
        parts.append(
            f"[{key}] {authors_str} ({article.year}). {article.title}\n   {article.abstract}\n"
        )
    return "\n".join(parts)


def _format_apa_reference(ref: Reference) -> str:
    """Format a single reference in APA style.

    Args:
        ref: Reference object.

    Returns:
        APA-formatted reference string.
    """
    if ref.authors:
        authors_str = ", ".join(ref.authors[:6])
        if len(ref.authors) > 6:
            authors_str += ", ... "
    else:
        authors_str = "Unknown"

    doi_str = f" https://doi.org/{ref.doi}" if ref.doi else ""

    return f"{authors_str} ({ref.year}). {ref.title}. *{ref.journal}*.{doi_str} PMID: {ref.pmid}"


def _assemble_markdown(draft: ReviewDraft) -> str:
    """Assemble the final Markdown review document.

    Args:
        draft: Complete ReviewDraft with all sections and references.

    Returns:
        Formatted Markdown string.
    """
    parts: list[str] = [f"# {draft.title}\n"]

    if draft.abstract:
        parts.append(f"## Abstract\n\n{draft.abstract}\n")

    for section in draft.sections:
        parts.append(f"## {section.heading}\n\n{section.body}\n")

    if draft.references:
        parts.append("## References\n")
        for ref in sorted(draft.references, key=lambda r: (r.year, r.key)):
            parts.append(f"- [{ref.key}] {_format_apa_reference(ref)}")

    return "\n".join(parts)
