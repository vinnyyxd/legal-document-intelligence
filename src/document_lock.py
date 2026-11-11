"""Per-document processing coordination for one Python process."""

from contextlib import contextmanager
from threading import Lock

_registry_lock = Lock()
_document_locks = {}


class DocumentAlreadyProcessingError(RuntimeError):
    """Raised when another operation holds a document lock."""


@contextmanager
def document_processing_lock(document_id):
    """Acquire a nonblocking, process-local lock for a document."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("A nonempty document ID is required.")

    with _registry_lock:
        lock = _document_locks.setdefault(document_id, Lock())

    if not lock.acquire(blocking=False):
        raise DocumentAlreadyProcessingError(
            "This document is already being processed. "
            "Wait for the current operation to finish."
        )

    try:
        yield
    finally:
        lock.release()
