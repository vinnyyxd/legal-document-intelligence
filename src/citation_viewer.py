"""
PDF citation viewer for Legal Document Intelligence.

Renders original PDF pages and highlights matching
evidence text when possible.

No changes to original PDFs, embeddings, or indexes.
"""

from pathlib import Path
from difflib import SequenceMatcher
import re

import fitz

from src.document_registry import (
    DEFAULT_STORAGE_ROOT,
    get_document,
)


def resolve_citation_pdf(
    document_id,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    """Resolve a PDF through its registered document identity."""

    document = get_document(
        document_id,
        storage_root=storage_root,
    )

    if document is None:
        raise ValueError(
            f"Document not found: {document_id}"
        )

    validated_id = document["document_id"]

    pdf_path = (
        Path(storage_root)
        / validated_id
        / "original.pdf"
    )

    if not pdf_path.is_file():
        raise FileNotFoundError(
            "Original PDF is unavailable."
        )

    return pdf_path


def _normalize_text(text):
    return " ".join(
        re.sub(r"\s+", " ", text).split()
    ).lower()


def _find_evidence_rectangles(page, evidence_text):
    """
    Locate matching evidence text on the PDF page.

    Uses exact phrase search first, then shorter
    distinctive phrases as a fallback.
    """

    if not evidence_text:
        return []

    text = str(evidence_text).strip()

    if not text:
        return []

    # Exact phrase search
    rectangles = page.search_for(text)

    if rectangles:
        return rectangles[:30]

    # Search shorter fragments of the evidence.
    words = text.split()

    if len(words) < 6:
        return []

    matches = []

    # Sample short consecutive phrases.
    # This accommodates line breaks and evidence
    # chunks spanning multiple PDF pages.
    for start in range(0, len(words), 12):

        phrase = " ".join(
            words[start:start + 8]
        )

        if len(phrase) < 20:
            continue

        found = page.search_for(phrase)

        if found:
            matches.extend(found)

        if len(matches) >= 30:
            break

    # Deduplicate rectangles
    unique = []
    seen = set()

    for rectangle in matches:
        key = tuple(
            round(value, 2)
            for value in rectangle
        )

        if key not in seen:
            seen.add(key)
            unique.append(rectangle)

    return unique[:30]


def render_citation_page(
    document_id,
    page_number,
    evidence_text="",
    storage_root=DEFAULT_STORAGE_ROOT,
    zoom=1.5,
):
    """
    Render a cited page from the original PDF.

    Returns:
        {
            "document_id": str,
            "page_number": int,
            "page_count": int,
            "image_bytes": bytes,
            "highlight_count": int,
        }
    """

    if not isinstance(page_number, int):
        raise ValueError(
            "Page number must be an integer."
        )

    if not 0.5 <= zoom <= 3.0:
        raise ValueError(
            "Zoom must be between 0.5 and 3.0."
        )

    pdf_path = resolve_citation_pdf(
        document_id,
        storage_root=storage_root,
    )

    with fitz.open(str(pdf_path)) as pdf:

        page_count = len(pdf)

        if not 1 <= page_number <= page_count:
            raise ValueError(
                f"Invalid page {page_number}. "
                f"Document contains {page_count} pages."
            )

        page = pdf[page_number - 1]

        # Search for matching evidence text.
        rectangles = _find_evidence_rectangles(
            page,
            evidence_text,
        )

        # Draw highlights onto the rendered page.
        # The source PDF remains unchanged.
        if rectangles:
            for rectangle in rectangles:
                page.draw_rect(
                    rectangle,
                    color=(1, 0.6, 0),
                    fill=(1, 0.9, 0.3),
                    fill_opacity=0.25,
                    overlay=True,
                )

        matrix = fitz.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image_bytes = pixmap.tobytes("png")

    return {
        "document_id": document_id,
        "page_number": page_number,
        "page_count": page_count,
        "image_bytes": image_bytes,
        "highlight_count": len(rectangles),
    }
