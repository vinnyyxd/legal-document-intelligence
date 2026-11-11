
import time
import re
import numpy as np

from pathlib import Path
from google.genai import types


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768


# --------------------------------------------------
# 1. Prepare legal text
# --------------------------------------------------

def prepare_chunk_text(chunk):

    parts = [
        f"Document: {chunk['document']}"
    ]

    if chunk.get("chapter_title"):
        parts.append(
            f"Chapter: {chunk['chapter_title']}"
        )

    if chunk.get("article_title"):
        parts.append(
            f"Article title: {chunk['article_title']}"
        )

    if chunk.get("clause_reference"):
        parts.append(
            f"Legal provision: {chunk['clause_reference']}"
        )

    parts.append(
        f"Legal text: {chunk['text']}"
    )

    return "\n".join(parts)


# --------------------------------------------------
# 2. Normalize embeddings
# --------------------------------------------------

def normalize_embeddings(vectors):

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    if not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding values.")

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    if np.any(norms == 0):
        raise ValueError("Zero-length embedding detected.")

    return vectors / norms


# --------------------------------------------------
# 3. Generate embeddings
# --------------------------------------------------

def embed_texts(
    client,
    texts,
    task_type="RETRIEVAL_DOCUMENT"
):

    if not texts:
        raise ValueError("No text supplied.")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    if not response.embeddings:
        raise ValueError("Gemini returned no embeddings.")

    if len(response.embeddings) != len(texts):
        raise ValueError("Embedding count mismatch.")

    vectors = [
        embedding.values
        for embedding in response.embeddings
    ]

    vectors = normalize_embeddings(vectors)

    expected_shape = (
        len(texts),
        EMBEDDING_DIMENSION
    )

    if vectors.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}, got {vectors.shape}"
        )

    return vectors


# --------------------------------------------------
# 4. Save checkpoint safely
# --------------------------------------------------

def save_checkpoint(checkpoint_path, completed):

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ids = list(completed.keys())

    vectors = np.vstack([
        completed[chunk_id] for chunk_id in ids
    ]) if ids else np.empty(
        (0, EMBEDDING_DIMENSION),
        dtype=np.float32
    )

    temp_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    # Write to temporary file before replacing checkpoint
    with open(temp_path, "wb") as f:

        np.savez_compressed(
            f,
            chunk_ids=np.asarray(ids, dtype=str),
            vectors=vectors,
            model=np.asarray(EMBEDDING_MODEL),
            dimension=np.asarray(EMBEDDING_DIMENSION)
        )

    temp_path.replace(checkpoint_path)


# --------------------------------------------------
# 5. Load previous checkpoint
# --------------------------------------------------

def load_checkpoint(checkpoint_path, expected_ids):

    checkpoint_path = Path(checkpoint_path)

    completed = {}

    if not checkpoint_path.exists():
        return completed

    with np.load(
        checkpoint_path,
        allow_pickle=False
    ) as saved:

        model = str(saved["model"].item())
        dimension = int(saved["dimension"].item())

        if model != EMBEDDING_MODEL:
            raise ValueError("Checkpoint model mismatch.")

        if dimension != EMBEDDING_DIMENSION:
            raise ValueError("Checkpoint dimension mismatch.")

        ids = saved["chunk_ids"].tolist()
        vectors = saved["vectors"]

    if vectors.shape != (len(ids), EMBEDDING_DIMENSION):
        raise ValueError("Invalid checkpoint vector shape.")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in checkpoint.")

    expected_set = set(expected_ids)

    if not set(ids).issubset(expected_set):
        raise ValueError(
            "Checkpoint contains unknown chunk IDs."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Checkpoint contains invalid vectors.")

    for chunk_id, vector in zip(ids, vectors):

        norm = np.linalg.norm(vector)

        if not np.isclose(norm, 1.0, atol=1e-4):
            raise ValueError(
                f"Invalid normalized vector: {chunk_id}"
            )

        completed[chunk_id] = vector

    return completed


# --------------------------------------------------
# 6. Generate embeddings for all legal chunks
# --------------------------------------------------

def embed_chunks(
    client,
    chunks,
    batch_size=8,
    checkpoint_path=None,
    pause_seconds=1.0,
    max_retries=5
):

    if not chunks:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32
        )

    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    if max_retries < 1:
        raise ValueError("At least one attempt is required.")

    chunk_ids = [
        chunk["chunk_id"] for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    # Restore previously generated vectors
    if checkpoint_path:

        completed = load_checkpoint(
            checkpoint_path,
            chunk_ids
        )

    else:
        completed = {}

    pending = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in completed
    ]

    print("Total chunks:", len(chunks))
    print("Already embedded:", len(completed))
    print("Remaining:", len(pending))

    # Process remaining chunks in batches
    for start in range(0, len(pending), batch_size):

        batch = pending[start:start + batch_size]

        texts = [
            prepare_chunk_text(chunk)
            for chunk in batch
        ]

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
                    for marker in [
                        "429", "500", "502", "503", "504",
                        "resource_exhausted", "unavailable",
                        "deadline", "timeout", "temporarily"
                    ]
                )

                if not retryable or attempt == max_retries - 1:
                    raise

                # Respect Google's RetryInfo when available
                match = re.search(
                    r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
                    str(exc)
                )

                if match:
                    provider_delay = float(match.group(1))

                    if provider_delay > 120:
                        raise RuntimeError(
                            "Gemini requested a retry delay of "
                            f"{provider_delay:.0f} seconds. "
                            "Legacy embedding processing stopped "
                            "to avoid an excessive wait. "
                            "Previously saved checkpoints remain "
                            "available for resume."
                        ) from exc

                    wait = provider_delay + 2
                else:
                    wait = min(120, 2 ** (attempt + 1))

                print(
                    f"Temporary API error. "
                    f"Retrying in {wait:.0f} seconds..."
                )

                time.sleep(wait)

        # Store successful results
        for chunk, vector in zip(batch, vectors):

            completed[chunk["chunk_id"]] = vector

        # Save progress after every successful batch
        if checkpoint_path:

            save_checkpoint(
                checkpoint_path,
                completed
            )

        print(
            f"Embedded {len(completed)}/{len(chunks)} chunks"
        )

        if pause_seconds > 0 and start + batch_size < len(pending):
            time.sleep(pause_seconds)

    # Return vectors in exactly the same order as input chunks
    result = np.vstack([
        completed[chunk_id]
        for chunk_id in chunk_ids
    ])

    assert result.shape == (
        len(chunks),
        EMBEDDING_DIMENSION
    )

    assert np.allclose(
        np.linalg.norm(result, axis=1),
        1.0,
        atol=1e-4
    )

    return result
