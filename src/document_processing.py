import json
from pathlib import Path

from src.extraction import extract_pdf
from src.chunking import create_legal_chunks, validate_chunks

from src.document_registry import (
    DEFAULT_STORAGE_ROOT,
    get_document,
    update_document_status,
    _atomic_write_json,
)


def process_document(
    document_id,
    storage_root=DEFAULT_STORAGE_ROOT,
    enable_ocr=False,
    ocr_language="eng",
):
    document = get_document(
        document_id,
        storage_root=storage_root
    )

    if document["status"] != "registered":
        raise ValueError(
            "Only registered documents can begin processing."
        )

    document_id = document["document_id"]
    filename = document["filename"]

    # Derive the storage directory from the validated UUID,
    # rather than trusting a path stored in registry metadata.
    document_directory = (
        Path(storage_root) / document_id
    )

    pdf_path = document_directory / "original.pdf"

    try:
        # ----------------------------------------
        # Stage 1: PDF extraction
        # ----------------------------------------

        update_document_status(
            document_id,
            "extracting",
            storage_root=storage_root
        )

        pages = extract_pdf(
            str(pdf_path),
            document_id=document_id,
            enable_ocr=enable_ocr,
            ocr_language=ocr_language,
        )

        if not pages:
            raise ValueError(
                "PDF extraction returned no pages."
            )

        # Prevent incomplete OCR evidence from being indexed.
        # Legacy extraction remains unchanged when OCR is disabled.
        if enable_ocr:
            incomplete_ocr_pages = [
                page
                for page in pages
                if page.get("ocr_status")
                in {"failed", "no_text"}
            ]

            if incomplete_ocr_pages:
                details = "; ".join(
                    "page "
                    + str(page["page"])
                    + ": "
                    + str(
                        page.get("ocr_error")
                        or page.get("ocr_status")
                    )
                    for page in incomplete_ocr_pages
                )

                raise ValueError(
                    "OCR could not recover text from "
                    "all attempted pages: "
                    + details
                )

        # Preserve the user's original filename.
        for page in pages:
            page["document"] = filename

        _atomic_write_json(
            document_directory / "extracted.json",
            pages
        )

        update_document_status(
            document_id,
            "extracted",
            storage_root=storage_root
        )

        # ----------------------------------------
        # Stage 2: Legal chunking
        # ----------------------------------------

        update_document_status(
            document_id,
            "chunking",
            storage_root=storage_root
        )

        chunks = create_legal_chunks(pages)

        validation = validate_chunks(chunks)

        if not validation["valid"]:
            raise ValueError(
                "Legal chunk validation failed: "
                + "; ".join(validation["errors"])
            )

        if not chunks:
            raise ValueError(
                "No legal chunks generated from PDF."
            )

        # Verify identity before persisting chunks.
        for chunk in chunks:
            if chunk.get("document_id") != document_id:
                raise ValueError(
                    "Chunk document identity mismatch."
                )

        _atomic_write_json(
            document_directory / "chunks.json",
            chunks
        )

        _atomic_write_json(
            document_directory / "processing_validation.json",
            validation
        )

        update_document_status(
            document_id,
            "chunked",
            storage_root=storage_root
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "status": "chunked",
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "validation": validation,
            "ocr_summary": (
                {
                    "enabled": True,
                    "attempted_pages": sum(
                        page.get("ocr_status")
                        in {"success", "no_text", "failed"}
                        for page in pages
                    ),
                    "successful_pages": sum(
                        page.get("ocr_status") == "success"
                        for page in pages
                    ),
                    "failed_pages": sum(
                        page.get("ocr_status") == "failed"
                        for page in pages
                    ),
                }
                if enable_ocr
                else None
            ),
        }

    except Exception as exc:

        # Failure transitions are allowed only
        # from active processing states.
        current = get_document(
            document_id,
            storage_root=storage_root
        )

        if current["status"] in {
            "extracting",
            "chunking"
        }:
            update_document_status(
                document_id,
                "failed",
                storage_root=storage_root,
                error=str(exc) or type(exc).__name__
            )

        raise
