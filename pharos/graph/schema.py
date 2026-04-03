"""Pydantic models for biomedical knowledge-graph entities and relations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    """Supported knowledge-graph node types."""

    GENE = "Gene"
    DISEASE = "Disease"
    DRUG = "Drug"
    PROTEIN = "Protein"
    PATHWAY = "Pathway"
    COMPOUND = "Compound"
    PHENOTYPE = "Phenotype"
    ANATOMY = "Anatomy"


class RelationType(StrEnum):
    """Supported knowledge-graph relation types."""

    TARGETS = "TARGETS"
    TREATS = "TREATS"
    CAUSES = "CAUSES"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    INTERACTS_WITH = "INTERACTS_WITH"
    PART_OF = "PART_OF"
    REGULATES = "REGULATES"
    EXPRESSED_IN = "EXPRESSED_IN"
    ENCODES = "ENCODES"
    INHIBITS = "INHIBITS"
    ACTIVATES = "ACTIVATES"


class BioEntity(BaseModel):
    """Base model for all biomedical entities."""

    node_type: NodeType
    name: str
    external_ids: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    source: str = ""


class Gene(BioEntity):
    """A gene entity."""

    node_type: NodeType = NodeType.GENE
    symbol: str = ""
    organism: str = "Homo sapiens"


class Disease(BioEntity):
    """A disease entity."""

    node_type: NodeType = NodeType.DISEASE
    mondo_id: str = ""
    icd10: str = ""


class Drug(BioEntity):
    """A drug / therapeutic entity."""

    node_type: NodeType = NodeType.DRUG
    drugbank_id: str = ""
    phase: str = ""


class Protein(BioEntity):
    """A protein entity."""

    node_type: NodeType = NodeType.PROTEIN
    uniprot_id: str = ""
    sequence: str = ""
    organism: str = "Homo sapiens"


class Pathway(BioEntity):
    """A biological pathway entity."""

    node_type: NodeType = NodeType.PATHWAY
    reactome_id: str = ""


class Compound(BioEntity):
    """A small-molecule compound entity."""

    node_type: NodeType = NodeType.COMPOUND
    smiles: str = ""
    inchi: str = ""
    molecular_weight: float = 0.0


class Relation(BaseModel):
    """A directed relation between two biomedical entities."""

    source_name: str
    source_type: NodeType
    target_name: str
    target_type: NodeType
    relation_type: RelationType
    confidence: float = 0.0
    source_ref: str = ""
    properties: dict[str, str] = Field(default_factory=dict)
