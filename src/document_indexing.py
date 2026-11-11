import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.embeddings import (
    EMBEDDING_DIMENSION,
)

from src.retrieval import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
)

from src.embedding_provenance import (
    validate_embedding_manifest,
)

from src.document_registry import (
    DEFAULT_STORAGE_ROOT,
    get_document,
    update_document_status,
    _atomic_write_json,
)


def index_document(
    document_id,
    vectors,
    embedding_chunk_ids,
    embedding_manifest,
    storage_root=DEFAULT_STORAGE_ROOT,
    embedding_model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
):
    document = get_document(
        document_id,
        storage_root=storage_root
    )

    if document["status"] != "chunked":
        raise ValueError(
            "Only chunked documents can be indexed."
        )

    document_id = document["document_id"]

    document_directory = (
        Path(storage_root) / document_id
    )

    chunks_path = document_directory / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            "Document chunks are missing."
        )

    import json

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    if not chunks:
        raise ValueError(
            "Document has no chunks."
        )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs detected."
        )

    if not all(
        chunk.get("document_id") == document_id
        for chunk in chunks
    ):
        raise ValueError(
            "Chunk document identity mismatch."
        )

    # Explicitly validate the vector-to-chunk row mapping.
    # Each embedding row must correspond to the chunk
    # at the same position in chunks.json.
    if not isinstance(embedding_chunk_ids, (list, tuple)):
        raise ValueError(
            "embedding_chunk_ids must be a list or tuple."
        )

    if not all(
        isinstance(chunk_id, str) and chunk_id
        for chunk_id in embedding_chunk_ids
    ):
        raise ValueError(
            "Embedding chunk IDs must be non-empty strings."
        )

    if len(embedding_chunk_ids) != len(chunks):
        raise ValueError(
            "Embedding chunk-ID count mismatch."
        )

    if len(set(embedding_chunk_ids)) != len(
        embedding_chunk_ids
    ):
        raise ValueError(
            "Duplicate embedding chunk IDs detected."
        )

    if list(embedding_chunk_ids) != chunk_ids:
        raise ValueError(
            "Embedding chunk-ID order does not match "
            "the document's stored chunk order."
        )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix."
        )

    if vectors.shape != (
        len(chunks),
        EMBEDDING_DIMENSION
    ):
        raise ValueError(
            "Embedding dimensions or chunk count mismatch."
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Embeddings contain invalid values."
        )

    # Embeddings must already be normalized by the
    # embedding-generation pipeline.
    #
    # Do not normalize again: even tiny float32 rounding
    # changes would invalidate exact SHA-256 fingerprints.
    norms = np.linalg.norm(
        vectors,
        axis=1
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4
    ):
        raise ValueError(
            "Embedding vectors must already be normalized."
        )

    # Validate provenance before advancing lifecycle.
    validate_embedding_manifest(
        embedding_manifest,
        chunks,
        vectors,
        expected_model=embedding_model,
        expected_task_type=task_type,
    )

    # All input checks have passed.
    update_document_status(
        document_id,
        "embedding",
        storage_root=storage_root
    )

    try:
        update_document_status(
            document_id,
            "embedded",
            storage_root=storage_root
        )

        update_document_status(
            document_id,
            "indexing",
            storage_root=storage_root
        )

        index = build_faiss_index(vectors)

        metadata = {
            "document_id": document_id,
            "model": embedding_model,
            "dimensions": EMBEDDING_DIMENSION,
            "normalized": True,
            "task_type": task_type,
            "total_chunks": len(chunks),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "rows": [
                {
                    "row_index": i,
                    "chunk_id": chunk["chunk_id"],
                    "document_id": document_id,
                }
                for i, chunk in enumerate(chunks)
            ],
        }

        # Stage files in the private document directory.
        # Publish metadata and vectors using atomic replacement.
        with tempfile.NamedTemporaryFile(
            dir=document_directory,
            prefix=".embeddings_",
            suffix=".npy",
            delete=False
        ) as temporary_file:
            temporary_vectors_path = Path(
                temporary_file.name
            )

        try:
            np.save(
                temporary_vectors_path,
                vectors,
                allow_pickle=False
            )

            os.replace(
                temporary_vectors_path,
                document_directory / "embeddings.npy"
            )

        finally:
            temporary_vectors_path.unlink(
                missing_ok=True
            )

        save_faiss_index(
            index,
            document_directory / "faiss.index"
        )

        _atomic_write_json(
            document_directory / "embedding_metadata.json",
            metadata
        )

        _atomic_write_json(
            document_directory / "embedding_provenance.json",
            embedding_manifest
        )

        # Confirm that the saved index is readable.
        restored_index = load_faiss_index(
            document_directory / "faiss.index"
        )

        if (
            restored_index.ntotal != len(chunks)
            or restored_index.d != EMBEDDING_DIMENSION
        ):
            raise ValueError(
                "Persisted FAISS index validation failed."
            )

        update_document_status(
            document_id,
            "ready",
            storage_root=storage_root
        )

        return {
            "document_id": document_id,
            "status": "ready",
            "chunk_count": len(chunks),
            "embedding_dimensions": EMBEDDING_DIMENSION,
            "index_size": restored_index.ntotal,
        }

    except Exception as exc:
        current = get_document(
            document_id,
            storage_root=storage_root
        )

        if current["status"] in {
            "embedding",
            "indexing"
        }:
            update_document_status(
                document_id,
                "failed",
                storage_root=storage_root,
                error=str(exc) or type(exc).__name__
            )

        raise
