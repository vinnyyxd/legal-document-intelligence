
import numpy as np

from unittest.mock import patch

from src.document_embedding import (
    embed_document_chunks,
    save_safe_checkpoint,
    _checkpoint_fingerprints,
)
from src.embeddings import EMBEDDING_DIMENSION


def make_chunks():
    return [
        {
            "chunk_id": f"progress_chunk_{i}",
            "document": "progress.pdf",
            "text": f"Legal paragraph {i}",
        }
        for i in range(3)
    ]


def make_vector(index):
    vector = np.zeros(
        EMBEDDING_DIMENSION,
        dtype=np.float32,
    )
    vector[index] = 1.0
    return vector


def test_progress_events_for_partial_checkpoint(tmp_path):
    chunks = make_chunks()
    checkpoint = tmp_path / "checkpoint.npz"

    save_safe_checkpoint(
        checkpoint,
        {chunks[0]["chunk_id"]: make_vector(0)},
        _checkpoint_fingerprints(chunks),
    )

    events = []

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
            progress_callback=events.append,
        )

    mocked.assert_called_once()

    assert [event["event"] for event in events] == [
        "checkpoint_loaded",
        "batch_started",
        "batch_saved",
        "embedding_complete",
    ]

    assert events[0]["reused"] == 1
    assert events[0]["completed"] == 1
    assert events[-1]["completed"] == 3
    assert events[-1]["total"] == 3

    assert result["reused_chunks"] == 1
    assert result["generated_chunks"] == 2


def test_retry_event_without_real_wait(tmp_path):
    chunks = make_chunks()[:1]
    checkpoint = tmp_path / "retry.npz"

    events = []

    with patch(
        "src.document_embedding.embed_texts",
        side_effect=[
            RuntimeError("429 resource_exhausted"),
            np.vstack([make_vector(0)]),
        ],
    ) as mocked, patch(
        "src.document_embedding.time.sleep"
    ) as mocked_sleep:

        result = embed_document_chunks(
            client=None,
            chunks=chunks,
            checkpoint_path=checkpoint,
            progress_callback=events.append,
        )

    assert mocked.call_count == 2
    mocked_sleep.assert_called_once()

    retry_events = [
        event
        for event in events
        if event["event"] == "retry_wait"
    ]

    assert len(retry_events) == 1
    assert retry_events[0]["attempt"] == 1
    assert retry_events[0]["wait_seconds"] > 0

    assert events[-1]["event"] == "embedding_complete"
    assert result["generated_chunks"] == 1


def test_complete_checkpoint_reports_no_generation(tmp_path):
    chunks = make_chunks()[:1]
    checkpoint = tmp_path / "complete.npz"

    save_safe_checkpoint(
        checkpoint,
        {chunks[0]["chunk_id"]: make_vector(0)},
        _checkpoint_fingerprints(chunks),
    )

    events = []

    with patch(
        "src.document_embedding.embed_texts",
        side_effect=AssertionError(
            "No embedding request expected"
        ),
    ) as mocked:
        result = embed_document_chunks(
            client=None,
            chunks=chunks,
            checkpoint_path=checkpoint,
            progress_callback=events.append,
        )

    mocked.assert_not_called()

    assert [event["event"] for event in events] == [
        "checkpoint_loaded",
        "embedding_complete",
    ]

    assert events[-1]["reused"] == 1
    assert events[-1]["generated"] == 0
    assert result["generated_chunks"] == 0
