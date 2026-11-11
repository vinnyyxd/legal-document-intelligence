import hashlib

import numpy as np

from src.embeddings import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    prepare_chunk_text,
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint_chunk(chunk):
    text = prepare_chunk_text(chunk)

    return _sha256(
        text.encode("utf-8")
    )


def fingerprint_vector(vector):
    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    if vector.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            "Invalid vector dimensions."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "Vector contains invalid values."
        )

    vector = np.ascontiguousarray(
        vector.astype("<f4", copy=False)
    )

    return _sha256(vector.tobytes())


def create_embedding_manifest(
    chunks,
    vectors,
    model=EMBEDDING_MODEL,
    task_type="RETRIEVAL_DOCUMENT",
):
    if not chunks:
        raise ValueError(
            "Cannot create manifest without chunks."
        )

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    if vectors.shape != (
        len(chunks),
        EMBEDDING_DIMENSION
    ):
        raise ValueError(
            "Chunk and embedding shape mismatch."
        )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs detected."
        )

    if not isinstance(model, str) or not model.strip():
        raise ValueError("Invalid embedding model.")

    if (
        not isinstance(task_type, str)
        or not task_type.strip()
    ):
        raise ValueError("Invalid embedding task type.")

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Invalid embedding values."
        )

    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4
    ):
        raise ValueError(
            "Embedding vectors must be normalized."
        )

    rows = []

    for index, (chunk, vector) in enumerate(
        zip(chunks, vectors)
    ):
        rows.append({
            "row_index": index,
            "chunk_id": chunk["chunk_id"],
            "text_sha256": fingerprint_chunk(chunk),
            "vector_sha256": fingerprint_vector(vector),
        })

    return {
        "schema_version": 1,
        "model": model,
        "dimensions": EMBEDDING_DIMENSION,
        "task_type": task_type,
        "normalized": True,
        "total_chunks": len(chunks),
        "rows": rows,
    }


def validate_embedding_manifest(
    manifest,
    chunks,
    vectors,
    expected_model=EMBEDDING_MODEL,
    expected_task_type="RETRIEVAL_DOCUMENT",
):
    if not isinstance(manifest, dict):
        raise ValueError(
            "Manifest must be a dictionary."
        )

    if manifest.get("schema_version") != 1:
        raise ValueError(
            "Unsupported manifest schema."
        )

    if manifest.get("model") != expected_model:
        raise ValueError(
            "Embedding model mismatch."
        )

    if manifest.get("task_type") != expected_task_type:
        raise ValueError(
            "Embedding task type mismatch."
        )

    if manifest.get("dimensions") != EMBEDDING_DIMENSION:
        raise ValueError(
            "Embedding dimension mismatch."
        )

    if manifest.get("normalized") is not True:
        raise ValueError(
            "Manifest normalization mismatch."
        )

    expected = create_embedding_manifest(
        chunks,
        vectors,
        model=expected_model,
        task_type=expected_task_type,
    )

    if manifest != expected:
        raise ValueError(
            "Embedding provenance mismatch: "
            "chunk text, vector content, "
            "or row alignment has changed."
        )

    return {
        "valid": True,
        "total_chunks": len(chunks),
        "model": expected_model,
        "task_type": expected_task_type,
    }
