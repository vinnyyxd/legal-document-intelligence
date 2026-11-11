
import pytest

from src.embedding_progress_ui import (
    make_embedding_progress_callback,
)


class FakeProgressBar:
    def __init__(self):
        self.calls = []

    def progress(self, fraction, text=None):
        self.calls.append({
            "fraction": fraction,
            "text": text,
        })


class FakeMessage:
    def __init__(self):
        self.calls = []

    def info(self, text):
        self.calls.append(("info", text))

    def warning(self, text):
        self.calls.append(("warning", text))

    def success(self, text):
        self.calls.append(("success", text))


def make_callback():
    bar = FakeProgressBar()
    message = FakeMessage()
    callback = make_embedding_progress_callback(bar, message)
    return callback, bar, message


def test_checkpoint_recovery_progress():
    callback, bar, message = make_callback()

    callback({
        "event": "checkpoint_loaded",
        "completed": 120,
        "total": 397,
        "reused": 120,
        "generated": 0,
    })

    assert bar.calls[-1]["fraction"] == pytest.approx(
        120 / 397
    )
    assert bar.calls[-1]["text"] == "Embeddings: 120/397"

    kind, text = message.calls[-1]
    assert kind == "info"
    assert "Recovered 120 saved embeddings" in text
    assert "277 remaining" in text


def test_batch_started_progress():
    callback, bar, message = make_callback()

    callback({
        "event": "batch_started",
        "completed": 8,
        "total": 16,
        "reused": 0,
        "generated": 8,
        "batch_size": 8,
    })

    assert bar.calls[-1]["fraction"] == pytest.approx(0.5)
    assert message.calls[-1][0] == "info"
    assert "8/16 completed" in message.calls[-1][1]


def test_retry_wait_progress():
    callback, bar, message = make_callback()

    callback({
        "event": "retry_wait",
        "completed": 8,
        "total": 16,
        "reused": 0,
        "generated": 8,
        "attempt": 2,
        "max_attempts": 6,
        "wait_seconds": 30,
    })

    assert bar.calls[-1]["fraction"] == pytest.approx(0.5)

    kind, text = message.calls[-1]
    assert kind == "warning"
    assert "Retry 2/6" in text
    assert "waiting 30 seconds" in text


def test_batch_saved_progress():
    callback, bar, message = make_callback()

    callback({
        "event": "batch_saved",
        "completed": 24,
        "total": 40,
        "reused": 8,
        "generated": 16,
    })

    assert bar.calls[-1]["fraction"] == pytest.approx(0.6)

    kind, text = message.calls[-1]
    assert kind == "info"
    assert "24/40 completed" in text
    assert "16 newly generated" in text


def test_embedding_complete_progress():
    callback, bar, message = make_callback()

    callback({
        "event": "embedding_complete",
        "completed": 40,
        "total": 40,
        "reused": 8,
        "generated": 32,
    })

    assert bar.calls[-1]["fraction"] == pytest.approx(1.0)

    kind, text = message.calls[-1]
    assert kind == "success"
    assert "40/40" in text
    assert "8 reused" in text
    assert "32 generated" in text


@pytest.mark.parametrize(
    "event",
    [
        None,
        "checkpoint_loaded",
        [],
        {"event": "unknown", "completed": 0, "total": 1},
        {"event": "batch_saved", "completed": 5, "total": 4},
        {"event": "batch_saved", "completed": -1, "total": 4},
        {"event": "batch_saved", "completed": "3", "total": 4},
    ],
)
def test_invalid_progress_events(event):
    callback, bar, message = make_callback()

    with pytest.raises((TypeError, ValueError)):
        callback(event)
