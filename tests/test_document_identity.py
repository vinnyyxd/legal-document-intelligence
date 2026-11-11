
import tempfile
from pathlib import Path

import fitz

from src.extraction import extract_pdf
from src.chunking import (
    split_into_paragraphs,
    create_legal_chunks,
    validate_chunks,
)


def make_page(document_id, article, text, filename="Contract.pdf"):
    return {
        "document": filename,
        "document_id": document_id,
        "page": 1,
        "text": (
            f"Article {article}\n"
            f"Test provision\n"
            f"1. {text}"
        ),
        "is_scanned": False,
        "word_count": len(text.split()),
    }


def test_same_filename_different_identity():

    pages = [
        make_page("doc_contract_001", 1, "Alpha obligations."),
        make_page("doc_contract_002", 4, "Beta obligations."),
    ]

    chunks = create_legal_chunks(pages)

    assert len(chunks) == 2
    assert len({c["chunk_id"] for c in chunks}) == 2

    assert chunks[0]["document_id"] == "doc_contract_001"
    assert chunks[1]["document_id"] == "doc_contract_002"

    assert chunks[0]["article"] == "1"
    assert chunks[1]["article"] == "4"

    assert validate_chunks(chunks)["valid"]


def test_hierarchy_isolation():

    pages = [
        make_page("doc_a", 8, "First document text."),
        {
            "document": "Contract.pdf",
            "document_id": "doc_b",
            "page": 1,
            "text": "1. Second document without article heading.",
            "is_scanned": False,
            "word_count": 6,
        },
    ]

    paragraphs = split_into_paragraphs(pages)

    assert len(paragraphs) == 2

    assert paragraphs[0]["article"] == "8"
    assert paragraphs[1]["article"] is None
    assert paragraphs[1]["document_id"] == "doc_b"


def test_sanitization_collision():

    pages = [
        make_page("doc:a", 1, "First document."),
        make_page("doc_a", 1, "Second document."),
    ]

    chunks = create_legal_chunks(pages)

    assert len(chunks) == 2

    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]

    assert chunks[0]["document_id"] == "doc:a"
    assert chunks[1]["document_id"] == "doc_a"


def test_legacy_chunk_compatibility():

    pages = [{
        "document": "gdpr.pdf",
        "page": 1,
        "text": (
            "Article 5\n"
            "Principles relating to processing\n"
            "1. Personal data shall be processed lawfully."
        ),
        "is_scanned": False,
        "word_count": 10,
    }]

    chunks = create_legal_chunks(pages)

    assert len(chunks) == 1
    assert "document_id" not in chunks[0]
    assert chunks[0]["chunk_id"].startswith("gdpr_article_5_")


def test_pdf_extraction_identity():

    with tempfile.TemporaryDirectory() as directory:

        pdf_path = Path(directory) / "Contract.pdf"

        document = fitz.open()
        page = document.new_page()

        page.insert_text(
            (72, 72),
            "Article 1\nTest provision\n1. Contract obligations."
        )

        document.save(str(pdf_path))
        document.close()

        identified = extract_pdf(
            str(pdf_path),
            document_id="doc_contract_001",
        )

        legacy = extract_pdf(str(pdf_path))

        assert len(identified) == 1
        assert len(legacy) == 1

        assert identified[0]["document_id"] == "doc_contract_001"
        assert "document_id" not in legacy[0]

        assert identified[0]["text"] == legacy[0]["text"]
        assert identified[0]["document"] == "Contract.pdf"
