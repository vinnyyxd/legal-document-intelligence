
import pytest
from unittest.mock import patch

from src.embeddings import embed_chunks


def test_legacy_excessive_retry_delay_fails_fast():
    chunks = [
        {
            "chunk_id": "legacy_retry_test_001",
            "document": "test.pdf",
            "text": "A sample legal provision.",
        }
    ]

    error = RuntimeError(
        "429 RESOURCE_EXHAUSTED "
        "retryDelay: 30543s"
    )

    with patch(
        "src.embeddings.embed_texts",
        side_effect=error,
    ) as mocked_embed, patch(
        "src.embeddings.time.sleep",
    ) as mocked_sleep:

        with pytest.raises(
            RuntimeError,
            match="excessive wait",
        ):
            embed_chunks(
                client=None,
                chunks=chunks,
                batch_size=1,
                pause_seconds=0,
            )

    mocked_embed.assert_called_once()
    mocked_sleep.assert_not_called()
