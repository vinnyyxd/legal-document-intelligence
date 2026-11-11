
import numpy as np
import pytest

from unittest.mock import patch

from src.document_embedding import (
    embed_document_chunks,
    load_safe_checkpoint,
    save_safe_checkpoint,
    _checkpoint_fingerprints,
)
from src.embeddings import EMBEDDING_DIMENSION


def make_chunks(count=3):
    return [
        {
            "chunk_id": f"test_chunk_{index}",
            "document": "test.pdf",
            "text": f"Legal provision number {index}",
        }
        for index in range(count)
    ]


def make_vector(index):
    vector = np.zeros(
        EMBEDDING_DIMENSION,
        dtype=np.float32,
    )
    vector[index] = 1.0
    return vector


def seed_checkpoint(path, chunks, count):
    completed = {
        chunk["chunk_id"]: make_vector(index)
        for index, chunk in enumerate(chunks[:count])
    }

    save_safe_checkpoint(
        path,
        completed,
        _checkpoint_fingerprints(chunks),
    )


def test_complete_checkpoint_skips_embedding(tmp_path):
    chunks = make_chunks()
    checkpoint = tmp_path / "checkpoint.npz"

    seed_checkpoint(checkpoint, chunks, len(chunks))

    with patch(
        "src.document_embedding.embed_texts",
        side_effect=AssertionError(
            "Embedding API must not be called"
        ),
    ) as mocked:
        result = embed_document_chunks(
            client=None,
            chunks=chunks,
            checkpoint_path=checkpoint,
        )

    mocked.assert_not_called()

    assert result["reused_chunks"] == 3
    assert result["generated_chunks"] == 0
    assert result["vectors"].shape == (
        3,
        EMBEDDING_DIMENSION,
    )


def test_partial_checkpoint_only_embeds_missing(tmp_path):
    chunks = make_chunks()
    checkpoint = tmp_path / "checkpoint.npz"

    seed_checkpoint(checkpoint, chunks, 1)

    generated = np.vstack([
        make_vector(1),
        make_vector(2),
    ])

    with patch(
        "src.document_embedding.embed_texts",
        return_value=generated,
    ) as mocked:
        result = embed_document_chunks(
            client=None,
            chunks=chunks,
            checkpoint_path=checkpoint,
            batch_size=8,
        )

    assert mocked.call_count == 1

    submitted_texts = mocked.call_args.args[1]
    assert len(submitted_texts) == 2

    assert result["reused_chunks"] == 1
    assert result["generated_chunks"] == 2

    restored = load_safe_checkpoint(
        checkpoint,
        chunks,
    )
    assert len(restored) == 3


def test_interruption_preserves_checkpoint(tmp_path):
    chunks = make_chunks()
    checkpoint = tmp_path / "checkpoint.npz"

    seed_checkpoint(checkpoint, chunks, 1)

    with patch(
        "src.document_embedding.embed_texts",
        side_effect=RuntimeError(
            "Simulated non-retryable interruption"
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="Simulated non-retryable interruption",
        ):
            embed_document_chunks(
                client=None,
                chunks=chunks,
                checkpoint_path=checkpoint,
                batch_size=1,
            )

    restored = load_safe_checkpoint(
        checkpoint,
        chunks,
    )

    assert len(restored) == 1
    assert chunks[0]["chunk_id"] in restored


def test_retry_completes_remaining_chunks(tmp_path):
    chunks = make_chunks()
    checkpoint = tmp_path / "checkpoint.npz"

    seed_checkpoint(checkpoint, chunks, 1)

    generated = np.vstack([
        make_vector(1),
        make_vector(2),
    ])

    with patch(
        "src.document_embedding.embed_texts",
        return_value=generated,
    ) as mocked:
        result = embed_document_chunks(
            client=None,
            chunks=chunks,
            checkpoint_path=checkpoint,
            batch_size=8,
        )

    assert mocked.call_count == 1
    assert result["reused_chunks"] == 1
    assert result["generated_chunks"] == 2

    restored = load_safe_checkpoint(
        checkpoint,
        chunks,
    )

    assert len(restored) == 3


def test_stale_checkpoint_rejected(tmp_path):
    chunks = make_chunks()
    checkpoint = tmp_path / "checkpoint.npz"

    seed_checkpoint(checkpoint, chunks, 1)

    modified_chunks = [
        dict(chunk)
        for chunk in chunks
    ]
    modified_chunks[0]["text"] = (
        "Materially changed legal provision"
    )

    with patch(
        "src.document_embedding.embed_texts",
        side_effect=AssertionError(
            "Stale checkpoint must fail before embedding"
        ),
    ) as mocked:
        with pytest.raises(
            ValueError,
            match="Stale checkpoint",
        ):
            embed_document_chunks(
                client=None,
                chunks=modified_chunks,
                checkpoint_path=checkpoint,
            )

    mocked.assert_not_called()
