"""Grobid-based PDF parsing for scientific papers.

TODO (Phase 2):
    - Implement Grobid client for PDF → TEI-XML conversion
    - Add section extraction (abstract, methods, results, discussion)
    - Support reference extraction and linking
    - Implement figure/table caption extraction
"""

from __future__ import annotations


async def parse_pdf(pdf_path: str) -> dict[str, str]:
    """Parse a scientific PDF into structured sections using Grobid.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with keys: title, abstract, sections (list), references (list).

    Raises:
        NotImplementedError: PDF parsing not yet implemented.
    """
    raise NotImplementedError(
        "PDF parsing will use a local Grobid server for TEI-XML extraction in Phase 2."
    )


async def extract_references(pdf_path: str) -> list[dict[str, str]]:
    """Extract bibliographic references from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of reference dicts with title, authors, journal, year, doi.

    Raises:
        NotImplementedError: Reference extraction not yet implemented.
    """
    raise NotImplementedError(
        "Reference extraction will use Grobid's reference parser in Phase 2."
    )
