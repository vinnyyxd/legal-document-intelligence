import re
import time
import os
import tempfile

from pathlib import Path

import numpy as np

from src.embeddings import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    embed_texts,
    prepare_chunk_text,
)

from src.embedding_provenance import (
    fingerprint_chunk,
    fingerprint_vector,
    create_embedding_manifest,
)


def _validate_chunks(chunks):
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Chunks must be a non-empty list.")

    ids = [chunk["chunk_id"] for chunk in chunks]

    if not all(
        isinstance(chunk_id, str) and chunk_id
        for chunk_id in ids
    ):
        raise ValueError("Invalid chunk ID.")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs.")

    return ids


def _checkpoint_fingerprints(chunks):
    return {
        chunk["chunk_id"]: fingerprint_chunk(chunk)
        for chunk in chunks
    }


def save_safe_checkpoint(
    checkpoint_path,
    completed,
    fingerprints,
):
    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ids = list(completed)

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate checkpoint IDs.")

    if not set(ids).issubset(fingerprints):
        raise ValueError("Unknown checkpoint IDs.")

    vectors = (
        np.vstack([completed[cid] for cid in ids])
        if ids else
        np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32
        )
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.shape != (len(ids), EMBEDDING_DIMENSION):
        raise ValueError("Invalid checkpoint vector shape.")

    if not np.isfinite(vectors).all():
        raise ValueError("Invalid checkpoint vectors.")

    if len(ids) and not np.allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-4
    ):
        raise ValueError("Checkpoint vectors not normalized.")

    fd, temporary_name = tempfile.mkstemp(
        dir=checkpoint_path.parent,
        prefix=".checkpoint_",
        suffix=".npz"
    )

    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray(2),
                chunk_ids=np.asarray(ids, dtype=str),
                text_hashes=np.asarray(
                    [fingerprints[cid] for cid in ids],
                    dtype=str
                ),
                vector_hashes=np.asarray(
                    [
                        fingerprint_vector(vector)
                        for vector in vectors
                    ],
                    dtype=str
                ),
                vectors=vectors,
                model=np.asarray(EMBEDDING_MODEL),
                dimensions=np.asarray(EMBEDDING_DIMENSION),
                task_type=np.asarray("RETRIEVAL_DOCUMENT"),
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_name, checkpoint_path)

    finally:
        Path(temporary_name).unlink(missing_ok=True)


def load_safe_checkpoint(
    checkpoint_path,
    chunks,
):
    ids = _validate_chunks(chunks)
    fingerprints = _checkpoint_fingerprints(chunks)

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        return {}

    with np.load(checkpoint_path, allow_pickle=False) as saved:
        required = {
            "schema_version",
            "chunk_ids",
            "text_hashes",
            "vector_hashes",
            "vectors",
            "model",
            "dimensions",
            "task_type",
        }

        if not required.issubset(saved.files):
            raise ValueError("Incomplete safe checkpoint.")

        schema = int(saved["schema_version"].item())
        model = str(saved["model"].item())
        dimensions = int(saved["dimensions"].item())
        task_type = str(saved["task_type"].item())

        saved_ids = saved["chunk_ids"].tolist()
        saved_hashes = saved["text_hashes"].tolist()
        saved_vector_hashes = saved["vector_hashes"].tolist()
        vectors = saved["vectors"]

    if schema != 2:
        raise ValueError("Unsupported checkpoint schema.")

    if model != EMBEDDING_MODEL:
        raise ValueError("Checkpoint model mismatch.")

    if dimensions != EMBEDDING_DIMENSION:
        raise ValueError("Checkpoint dimension mismatch.")

    if task_type != "RETRIEVAL_DOCUMENT":
        raise ValueError("Checkpoint task mismatch.")

    if len(saved_ids) != len(set(saved_ids)):
        raise ValueError("Duplicate checkpoint IDs.")

    if not set(saved_ids).issubset(set(ids)):
        raise ValueError("Checkpoint contains unknown IDs.")

    if len(saved_hashes) != len(saved_ids):
        raise ValueError("Checkpoint fingerprint count mismatch.")

    if vectors.shape != (
        len(saved_ids),
        EMBEDDING_DIMENSION
    ):
        raise ValueError("Checkpoint vector shape mismatch.")

    if not np.isfinite(vectors).all():
        raise ValueError("Checkpoint contains invalid vectors.")

    if len(saved_ids) and not np.allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-4
    ):
        raise ValueError("Checkpoint vectors not normalized.")

    if len(saved_vector_hashes) != len(saved_ids):
        raise ValueError(
            "Checkpoint vector fingerprint count mismatch."
        )

    for vector, saved_vector_hash in zip(
        vectors,
        saved_vector_hashes
    ):
        if fingerprint_vector(vector) != saved_vector_hash:
            raise ValueError(
                "Checkpoint vector fingerprint mismatch."
            )

    for chunk_id, saved_hash in zip(
        saved_ids,
        saved_hashes
    ):
        if fingerprints[chunk_id] != saved_hash:
            raise ValueError(
                "Stale checkpoint: embedding text changed "
                f"for chunk {chunk_id}."
            )

    return {
        chunk_id: vector
        for chunk_id, vector in zip(saved_ids, vectors)
    }


def embed_document_chunks(
    client,
    chunks,
    checkpoint_path,
    batch_size=8,
    progress_callback=None,
):
    ids = _validate_chunks(chunks)

    if (
        progress_callback is not None
        and not callable(progress_callback)
    ):
        raise TypeError("progress_callback must be callable.")

    def emit(event, **details):
        if progress_callback is not None:
            progress_callback({
                "event": event,
                **details,
            })

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be positive.")

    fingerprints = _checkpoint_fingerprints(chunks)

    completed = load_safe_checkpoint(
        checkpoint_path,
        chunks
    )

    pending = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in completed
    ]

    reused = len(completed)
    total = len(chunks)

    emit(
        "checkpoint_loaded",
        completed=reused,
        total=total,
        reused=reused,
        generated=0,
    )

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]

        emit(
            "batch_started",
            completed=len(completed),
            total=total,
            reused=reused,
            generated=len(completed) - reused,
            batch_size=len(batch),
        )

        texts = [
            prepare_chunk_text(chunk)
            for chunk in batch
        ]

        # Retry temporary provider errors without losing
        # previously checkpointed embedding batches.
        max_retries = 6

        for attempt in range(max_retries):
            try:
                vectors = embed_texts(
                    client,
                    texts,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                break

            except Exception as exc:
                message = str(exc).lower()

                retryable = any(
                    marker in message
                    for marker in (
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "resource_exhausted",
                        "unavailable",
                        "deadline",
                        "timeout",
                        "temporarily",
                    )
                )

                if not retryable or attempt == max_retries - 1:
                    raise

                # Respect retry delays returned by Gemini.
                match = re.search(
                    '[\'"]?retryDelay[\'"]?\\s*:\\s*[\'"]?(\\d+(?:\\.\\d+)?)s',
                    str(exc),
                )

                if match:
                    provider_delay = float(match.group(1))

                    if provider_delay > 120:
                        raise RuntimeError(
                            "Gemini requested a retry delay of "
                            f"{provider_delay:.0f} seconds. "
                            "Processing stopped to avoid an excessive "
                            "wait. Previously saved embedding "
                            "checkpoints remain available for resume."
                        ) from exc

                    wait_seconds = provider_delay + 2
                else:
                    wait_seconds = min(
                        120,
                        2 ** (attempt + 1),
                    )

                emit(
                    "retry_wait",
                    completed=len(completed),
                    total=total,
                    reused=reused,
                    generated=len(completed) - reused,
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    wait_seconds=wait_seconds,
                )

                print(
                    "Temporary embedding API error. "
                    f"Retrying in {wait_seconds:.0f} seconds..."
                )

                time.sleep(wait_seconds)

        # Validate the complete provider response before
        # modifying completed results or the checkpoint.
        vectors = np.asarray(vectors, dtype=np.float32)

        expected_shape = (
            len(batch),
            EMBEDDING_DIMENSION
        )

        if vectors.shape != expected_shape:
            raise ValueError(
                "Embedding batch shape mismatch: "
                f"expected {expected_shape}, "
                f"received {vectors.shape}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embedding batch contains non-finite values."
            )

        norms = np.linalg.norm(vectors, axis=1)

        if not np.allclose(
            norms,
            1.0,
            atol=1e-4
        ):
            raise ValueError(
                "Embedding batch contains unnormalized vectors."
            )

        for chunk, vector in zip(batch, vectors):
            completed[chunk["chunk_id"]] = vector

        save_safe_checkpoint(
            checkpoint_path,
            completed,
            fingerprints
        )

        emit(
            "batch_saved",
            completed=len(completed),
            total=total,
            reused=reused,
            generated=len(completed) - reused,
        )

    result = np.vstack([
        completed[chunk_id]
        for chunk_id in ids
    ])

    manifest = create_embedding_manifest(
        chunks,
        result,
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT"
    )

    emit(
        "embedding_complete",
        completed=total,
        total=total,
        reused=reused,
        generated=len(pending),
    )

    return {
        "vectors": result,
        "chunk_ids": ids,
        "manifest": manifest,
        "reused_chunks": len(chunks) - len(pending),
        "generated_chunks": len(pending),
    }
