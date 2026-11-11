
import json
from unittest.mock import patch

from src.document_pipeline import (
    ingest_document,
    run_document_pipeline,
)
from src.document_registry import (
    register_document,
    update_document_status,
)


def prepare_document(tmp_path):
    document = register_document(
        filename="progress.pdf",
        pdf_bytes=b"%PDF-1.4 mocked content",
        storage_root=tmp_path,
    )

    document_id = document["document_id"]

    for status in [
        "extracting",
        "extracted",
        "chunking",
        "chunked",
    ]:
        update_document_status(
            document_id,
            status,
            storage_root=tmp_path,
        )

    directory = tmp_path / document_id

    (directory / "chunks.json").write_text(
        json.dumps([
            {
                "chunk_id": "progress_chunk_001",
                "document_id": document_id,
                "document": "progress.pdf",
                "text": "Mock legal provision.",
            }
        ]),
        encoding="utf-8",
    )

    return document_id


def test_resume_forwards_progress_callback(tmp_path):
    document_id = prepare_document(tmp_path)
    events = []
    progress_callback = events.append

    def fake_embedding(**kwargs):
        callback = kwargs["progress_callback"]
        assert callback is progress_callback

        callback({
            "event": "checkpoint_loaded",
            "completed": 1,
            "total": 1,
            "reused": 1,
            "generated": 0,
        })

        return {
            "vectors": "mock_vectors",
            "chunk_ids": ["progress_chunk_001"],
            "manifest": "mock_manifest",
            "reused_chunks": 1,
            "generated_chunks": 0,
        }

    with patch(
        "src.document_pipeline.embed_document_chunks",
        side_effect=fake_embedding,
    ) as mocked_embedding, patch(
        "src.document_pipeline.index_document",
        return_value={"status": "ready"},
    ) as mocked_indexing:

        result = run_document_pipeline(
            client=None,
            document_id=document_id,
            storage_root=tmp_path,
            progress_callback=progress_callback,
        )

    mocked_embedding.assert_called_once()
    mocked_indexing.assert_called_once()

    assert len(events) == 1
    assert events[0]["event"] == "checkpoint_loaded"
    assert result["reused_chunks"] == 1


def test_ingestion_forwards_progress_callback(tmp_path):
    events = []
    progress_callback = events.append

    with patch(
        "src.document_pipeline.register_document",
        return_value={"document_id": "mock_doc_001"},
    ) as mocked_register, patch(
        "src.document_pipeline.run_document_pipeline",
        return_value={"status": "ready"},
    ) as mocked_pipeline:

        result = ingest_document(
            client=None,
            filename="uploaded.pdf",
            pdf_bytes=b"mock bytes",
            storage_root=tmp_path,
            progress_callback=progress_callback,
        )

    mocked_register.assert_called_once()
    mocked_pipeline.assert_called_once()

    forwarded = mocked_pipeline.call_args.kwargs

    assert forwarded["progress_callback"] is progress_callback
    assert forwarded["document_id"] == "mock_doc_001"
    assert result["status"] == "ready"
