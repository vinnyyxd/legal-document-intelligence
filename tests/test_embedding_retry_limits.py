
import numpy as np
import pytest

from unittest.mock import patch

from src.document_embedding import embed_document_chunks
from src.embeddings import EMBEDDING_DIMENSION


def make_chunks():
    return [
        {
            "chunk_id": "retry_test_chunk_001",
            "document": "test.pdf",
            "document_id": "retry_test_doc",
            "text": "A sample legal provision.",
        }
    ]


def fake_vector():
    vector = np.zeros(
        (1, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )
    vector[0, 0] = 1.0
    return vector


def test_excessive_retry_delay_stops_without_sleep(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"

    error = RuntimeError(
        "429 RESOURCE_EXHAUSTED "
        "retryDelay: 30543s"
    )

    with patch(
        "src.document_embedding.embed_texts",
        side_effect=error,
    ) as mocked_embed, patch(
        "src.document_embedding.time.sleep",
    ) as mocked_sleep:

        with pytest.raises(
            RuntimeError,
            match="excessive wait",
        ):
            embed_document_chunks(
                client=None,
                chunks=make_chunks(),
                checkpoint_path=checkpoint,
            )

    mocked_embed.assert_called_once()
    mocked_sleep.assert_not_called()


def test_normal_provider_delay_is_respected(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"

    error = RuntimeError(
        "429 RESOURCE_EXHAUSTED "
        "retryDelay: 10s"
    )

    with patch(
        "src.document_embedding.embed_texts",
        side_effect=[error, fake_vector()],
    ) as mocked_embed, patch(
        "src.document_embedding.time.sleep",
    ) as mocked_sleep:

        result = embed_document_chunks(
            client=None,
            chunks=make_chunks(),
            checkpoint_path=checkpoint,
        )

    assert mocked_embed.call_count == 2
    mocked_sleep.assert_called_once_with(12.0)
    assert result["generated_chunks"] == 1


def test_missing_provider_delay_uses_backoff(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"

    error = RuntimeError(
        "503 Service temporarily unavailable"
    )

    with patch(
        "src.document_embedding.embed_texts",
        side_effect=[error, fake_vector()],
    ) as mocked_embed, patch(
        "src.document_embedding.time.sleep",
    ) as mocked_sleep:

        result = embed_document_chunks(
            client=None,
            chunks=make_chunks(),
            checkpoint_path=checkpoint,
        )

    assert mocked_embed.call_count == 2
    mocked_sleep.assert_called_once_with(2)
    assert result["generated_chunks"] == 1
