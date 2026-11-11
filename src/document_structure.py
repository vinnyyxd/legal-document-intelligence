"""
Document structure analysis for Legal Document Intelligence.

Uses persisted extraction metadata and PyMuPDF table detection.
Does not modify existing extraction, chunking, or RAG indexes.
"""

import json
from pathlib import Path

import fitz

from src.chunking import detect_legal_structure
from src.document_registry import (
    DEFAULT_STORAGE_ROOT,
    get_document,
)


def analyze_document_structure(
    document_id,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    """
    Analyze an already-ingested PDF.

    Returns page statistics, legal hierarchy,
    scanned-page warnings, and table metadata.
    """

    # Resolve the document through the existing registry.
    document = get_document(
        document_id,
        storage_root=storage_root,
    )

    if document is None:
        raise ValueError(
            f"Document not found: {document_id}"
        )

    # Use the validated document ID from the registry.
    document_id = document["document_id"]

    document_directory = (
        Path(storage_root) / document_id
    )

    extracted_path = (
        document_directory / "extracted.json"
    )

    pdf_path = (
        document_directory / "original.pdf"
    )

    if not extracted_path.is_file():
        raise FileNotFoundError(
            "Extracted page metadata is unavailable. "
            "Process the document first."
        )

    if not pdf_path.is_file():
        raise FileNotFoundError(
            "Original PDF is unavailable."
        )

    # --------------------------------------------------
    # 1. Load existing extraction metadata
    # --------------------------------------------------

    with extracted_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        pages = json.load(file)

    if not isinstance(pages, list):
        raise ValueError(
            "Invalid extracted page metadata."
        )

    page_count = len(pages)

    scanned_pages = sorted({
        page["page"]
        for page in pages
        if page.get("is_scanned", False)
    })

    readable_pages = (
        page_count - len(scanned_pages)
    )

    total_words = sum(
        page.get("word_count", 0)
        for page in pages
    )

    # --------------------------------------------------
    # 2. Detect legal hierarchy
    # --------------------------------------------------

    structure = detect_legal_structure(pages)

    chapters = []
    sections = []
    articles = []

    for item in structure:
        item_type = item.get("type")

        if item_type == "chapter":
            chapters.append(item)

        elif item_type == "section":
            sections.append(item)

        elif item_type == "article":
            articles.append(item)

    # --------------------------------------------------
    # 3. Detect PDF tables
    # --------------------------------------------------

    tables = []
    table_detection_warnings = []

    with fitz.open(str(pdf_path)) as pdf:

        if len(pdf) != page_count:
            raise ValueError(
                "PDF page count does not match extraction metadata."
            )

        for page_index in range(len(pdf)):

            page = pdf[page_index]

            try:
                detected = page.find_tables()

                for table_index, table in enumerate(
                    detected.tables,
                    start=1,
                ):
                    tables.append({
                        "page": page_index + 1,
                        "table_number": table_index,
                        "rows": table.row_count,
                        "columns": table.col_count,
                        "bbox": list(table.bbox),
                    })

            except Exception as exc:
                table_detection_warnings.append({
                    "page": page_index + 1,
                    "error": str(exc),
                })

    # --------------------------------------------------
    # 4. Return structure summary
    # --------------------------------------------------

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "status": document.get("status"),
        "page_count": page_count,
        "readable_page_count": readable_pages,
        "scanned_page_count": len(scanned_pages),
        "scanned_pages": scanned_pages,
        "total_words": total_words,
        "chapter_count": len(chapters),
        "section_count": len(sections),
        "article_count": len(articles),
        "table_count": len(tables),
        "chapters": chapters,
        "sections": sections,
        "articles": articles,
        "tables": tables,
        "table_detection_warnings": table_detection_warnings,
    }
