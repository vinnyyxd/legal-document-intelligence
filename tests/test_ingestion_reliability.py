
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.document_registry import (
    register_document,
    get_document,
    update_document_status,
)
from src.document_processing import process_document
from src.document_pipeline import run_document_pipeline


def register_test_document(tmp_path):

    return register_document(
        filename="Contract.pdf",
        pdf_bytes=b"%PDF-1.4\nsynthetic-test",
        storage_root=tmp_path,
    )


def page(document_id):

    return {
        "document": "Contract.pdf",
        "document_id": document_id,
        "page": 1,
        "text": "Article 1\nTest provision\n1. Obligations apply.",
        "is_scanned": False,
        "word_count": 7,
    }


def chunk(document_id):

    return {
        "chunk_id": f"{document_id}_chunk_1",
        "document_id": document_id,
        "document": "Contract.pdf",
        "text": "Obligations apply.",
        "page_numbers": [1],
    }


def test_extraction_failure_marks_failed(tmp_path):

    document = register_test_document(tmp_path)
    document_id = document["document_id"]

    with patch(
        "src.document_processing.extract_pdf",
        side_effect=RuntimeError("Synthetic extraction failure"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Synthetic extraction failure",
        ):
            process_document(
                document_id,
                storage_root=tmp_path,
            )

    stored = get_document(document_id, storage_root=tmp_path)

    assert stored["status"] == "failed"
    assert "Synthetic extraction failure" in stored["error"]


def test_empty_extraction_marks_failed(tmp_path):

    document = register_test_document(tmp_path)
    document_id = document["document_id"]

    with patch(
        "src.document_processing.extract_pdf",
        return_value=[],
    ):
        with pytest.raises(
            ValueError,
            match="no pages",
        ):
            process_document(
                document_id,
                storage_root=tmp_path,
            )

    assert get_document(
        document_id,
        storage_root=tmp_path,
    )["status"] == "failed"


def test_incomplete_ocr_rejected(tmp_path):

    document = register_test_document(tmp_path)
    document_id = document["document_id"]

    failed_page = page(document_id)
    failed_page["ocr_status"] = "failed"
    failed_page["ocr_error"] = "OCR engine unavailable"

    with patch(
        "src.document_processing.extract_pdf",
        return_value=[failed_page],
    ):
        with pytest.raises(
            ValueError,
            match="OCR could not recover",
        ):
            process_document(
                document_id,
                storage_root=tmp_path,
                enable_ocr=True,
            )

    stored = get_document(document_id, storage_root=tmp_path)

    assert stored["status"] == "failed"

    assert not (
        Path(tmp_path) / document_id / "chunks.json"
    ).exists()


def test_invalid_chunks_rejected(tmp_path):

    document = register_test_document(tmp_path)
    document_id = document["document_id"]

    with (
        patch(
            "src.document_processing.extract_pdf",
            return_value=[page(document_id)],
        ),
        patch(
            "src.document_processing.create_legal_chunks",
            return_value=[chunk(document_id)],
        ),
        patch(
            "src.document_processing.validate_chunks",
            return_value={
                "valid": False,
                "errors": ["Synthetic invalid chunk"],
            },
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Legal chunk validation failed",
        ):
            process_document(
                document_id,
                storage_root=tmp_path,
            )

    assert get_document(
        document_id,
        storage_root=tmp_path,
    )["status"] == "failed"


def test_chunk_identity_mismatch_rejected(tmp_path):

    document = register_test_document(tmp_path)
    document_id = document["document_id"]

    wrong_chunk = chunk("another_document")

    with (
        patch(
            "src.document_processing.extract_pdf",
            return_value=[page(document_id)],
        ),
        patch(
            "src.document_processing.create_legal_chunks",
            return_value=[wrong_chunk],
        ),
        patch(
            "src.document_processing.validate_chunks",
            return_value={
                "valid": True,
                "errors": [],
            },
        ),
    ):
        with pytest.raises(
            ValueError,
            match="identity mismatch",
        ):
            process_document(
                document_id,
                storage_root=tmp_path,
            )

    assert get_document(
        document_id,
        storage_root=tmp_path,
    )["status"] == "failed"

    assert not (
        Path(tmp_path) / document_id / "chunks.json"
    ).exists()


def prepare_chunked_document(tmp_path, chunks=None):

    document = register_test_document(tmp_path)
    document_id = document["document_id"]

    update_document_status(
        document_id,
        "extracting",
        storage_root=tmp_path,
    )
    update_document_status(
        document_id,
        "extracted",
        storage_root=tmp_path,
    )
    update_document_status(
        document_id,
        "chunking",
        storage_root=tmp_path,
    )
    update_document_status(
        document_id,
        "chunked",
        storage_root=tmp_path,
    )

    if chunks is not None:
        chunk_path = (
            Path(tmp_path) / document_id / "chunks.json"
        )
        chunk_path.write_text(
            json.dumps(chunks),
            encoding="utf-8",
        )

    return document_id


def test_missing_chunks_blocks_embedding(tmp_path):

    document_id = prepare_chunked_document(tmp_path)

    with patch(
        "src.document_pipeline.embed_document_chunks"
    ) as mocked_embedding:

        with pytest.raises(
            FileNotFoundError,
            match="chunks are missing",
        ):
            run_document_pipeline(
                client=None,
                document_id=document_id,
                storage_root=tmp_path,
            )

        mocked_embedding.assert_not_called()


def test_stored_identity_mismatch_blocks_embedding(tmp_path):

    document_id = prepare_chunked_document(
        tmp_path,
        chunks=[chunk("wrong_document")],
    )

    with patch(
        "src.document_pipeline.embed_document_chunks"
    ) as mocked_embedding:

        with pytest.raises(
            ValueError,
            match="mismatched document identity",
        ):
            run_document_pipeline(
                client=None,
                document_id=document_id,
                storage_root=tmp_path,
            )

        mocked_embedding.assert_not_called()
