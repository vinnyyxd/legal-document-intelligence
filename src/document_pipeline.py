import json
from pathlib import Path

from src.document_registry import (
    DEFAULT_STORAGE_ROOT,
    register_document,
    get_document,
)

from src.document_processing import process_document
from src.document_embedding import embed_document_chunks
from src.document_indexing import index_document


def run_document_pipeline(
    client,
    document_id,
    storage_root=DEFAULT_STORAGE_ROOT,
    batch_size=8,
    enable_ocr=False,
    ocr_language="eng",
):
    """
    Process a registered document through extraction,
    chunking, embedding, and FAISS indexing.

    A document already in 'chunked' status may resume
    embedding from its checkpoint.

    This function does not retry failed/ready documents.
    """

    document = get_document(
        document_id,
        storage_root=storage_root
    )

    status = document["status"]

    if status == "registered":
        process_document(
            document_id,
            storage_root=storage_root,
            enable_ocr=enable_ocr,
            ocr_language=ocr_language,
        )

    elif status != "chunked":
        raise ValueError(
            "Pipeline requires a registered or chunked "
            f"document; current status: {status}"
        )

    # Re-fetch canonical identity and current state.
    document = get_document(
        document_id,
        storage_root=storage_root
    )

    if document["status"] != "chunked":
        raise ValueError(
            "Document must be chunked before embedding."
        )

    document_id = document["document_id"]

    document_directory = (
        Path(storage_root) / document_id
    )

    chunks_path = document_directory / "chunks.json"

    if not chunks_path.is_file():
        raise FileNotFoundError(
            "Document chunks are missing."
        )

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "Stored document chunks must be a non-empty list."
        )

    if any(
        chunk.get("document_id") != document_id
        for chunk in chunks
    ):
        raise ValueError(
            "Stored chunks have mismatched document identity."
        )

    checkpoint_path = (
        document_directory / "embedding_checkpoint_v2.npz"
    )

    # This stage leaves status as 'chunked' until
    # embeddings and their provenance are complete.
    # Therefore, interrupted embedding can resume.
    embedding_result = embed_document_chunks(
        client=client,
        chunks=chunks,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
    )

    # The indexing module validates chunk order,
    # vector dimensions, and the provenance manifest
    # before advancing lifecycle state.
    indexing_result = index_document(
        document_id=document_id,
        vectors=embedding_result["vectors"],
        embedding_chunk_ids=embedding_result["chunk_ids"],
        embedding_manifest=embedding_result["manifest"],
        storage_root=storage_root,
    )

    return {
        **indexing_result,
        "filename": document["filename"],
        "reused_chunks": embedding_result["reused_chunks"],
        "generated_chunks": embedding_result["generated_chunks"],
        "checkpoint_path": str(checkpoint_path),
    }


def ingest_document(
    client,
    filename,
    pdf_bytes,
    storage_root=DEFAULT_STORAGE_ROOT,
    batch_size=8,
    enable_ocr=False,
    ocr_language="eng",
):
    """
    Register a PDF and execute the document pipeline.

    If processing raises, the registered document and
    any successfully written artifacts remain available
    for inspection.
    """

    document = register_document(
        filename=filename,
        pdf_bytes=pdf_bytes,
        storage_root=storage_root,
    )

    return run_document_pipeline(
        client=client,
        document_id=document["document_id"],
        storage_root=storage_root,
        batch_size=batch_size,
        enable_ocr=enable_ocr,
        ocr_language=ocr_language,
    )
