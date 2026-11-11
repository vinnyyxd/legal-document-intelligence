import json
import os
import tempfile
import uuid

from datetime import datetime, timezone
from pathlib import Path


DEFAULT_STORAGE_ROOT = Path("data/user_documents")


def _validate_document_id(document_id):
    try:
        parsed = uuid.UUID(str(document_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid document ID.")

    if str(parsed) != str(document_id):
        raise ValueError("Document ID must be canonical UUID format.")

    return str(parsed)


def _validate_filename(filename):
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")

    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise ValueError("Invalid filename.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF filenames are supported.")

    return filename


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".registry_",
            suffix=".tmp",
            delete=False
        ) as temporary_file:

            temporary_path = Path(temporary_file.name)

            json.dump(payload, temporary_file, indent=2)

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_registry(storage_root=DEFAULT_STORAGE_ROOT):
    registry_path = Path(storage_root) / "registry.json"

    if not registry_path.exists():
        return {"schema_version": 1, "documents": {}}

    with registry_path.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != 1
        or not isinstance(registry.get("documents"), dict)
    ):
        raise ValueError("Invalid document registry format.")

    return registry


def register_document(
    filename,
    pdf_bytes,
    storage_root=DEFAULT_STORAGE_ROOT
):
    filename = _validate_filename(filename)

    if not isinstance(pdf_bytes, bytes):
        raise TypeError("PDF content must be bytes.")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Uploaded content is not a valid PDF header.")

    if len(pdf_bytes) == 0:
        raise ValueError("Empty PDF content.")

    storage_root = Path(storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())

    document_directory = storage_root / document_id
    document_directory.mkdir(
        parents=False,
        exist_ok=False
    )

    pdf_path = document_directory / "original.pdf"

    try:
        with pdf_path.open("xb") as file:
            file.write(pdf_bytes)

        document = {
            "document_id": document_id,
            "filename": filename,
            "status": "registered",
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "pdf_path": str(pdf_path),
            "file_size": len(pdf_bytes)
        }

        registry = load_registry(storage_root)

        if document_id in registry["documents"]:
            raise ValueError("Document ID collision.")

        registry["documents"][document_id] = document

        _atomic_write_json(
            storage_root / "registry.json",
            registry
        )

        return document.copy()

    except Exception:
        pdf_path.unlink(missing_ok=True)

        try:
            document_directory.rmdir()
        except OSError:
            pass

        raise


def get_document(
    document_id,
    storage_root=DEFAULT_STORAGE_ROOT
):
    document_id = _validate_document_id(document_id)

    registry = load_registry(storage_root)

    document = registry["documents"].get(document_id)

    if document is None:
        raise KeyError(f"Document not found: {document_id}")

    return document.copy()


def list_documents(storage_root=DEFAULT_STORAGE_ROOT):
    registry = load_registry(storage_root)

    documents = [
        document.copy()
        for document in registry["documents"].values()
    ]

    return sorted(
        documents,
        key=lambda item: item["created_at"],
        reverse=True
    )


# --------------------------------------------------
# Document processing lifecycle
# --------------------------------------------------

DOCUMENT_TRANSITIONS = {
    "registered": {"extracting"},
    "extracting": {"extracted", "failed"},
    "extracted": {"chunking"},
    "chunking": {"chunked", "failed"},
    "chunked": {"embedding"},
    "embedding": {"embedded", "failed"},
    "embedded": {"indexing"},
    "indexing": {"ready", "failed"},
    "ready": set(),
    "failed": set(),
}


def update_document_status(
    document_id,
    new_status,
    storage_root=DEFAULT_STORAGE_ROOT,
    error=None
):
    document_id = _validate_document_id(document_id)

    registry = load_registry(storage_root)

    document = registry["documents"].get(document_id)

    if document is None:
        raise KeyError(f"Document not found: {document_id}")

    current_status = document["status"]

    if new_status not in DOCUMENT_TRANSITIONS.get(
        current_status, set()
    ):
        raise ValueError(
            f"Invalid status transition: "
            f"{current_status} -> {new_status}"
        )

    if error is not None and new_status != "failed":
        raise ValueError(
            "Processing errors can only be recorded "
            "when status is failed."
        )

    if new_status == "failed":
        if not isinstance(error, str) or not error.strip():
            raise ValueError(
                "A failure reason is required."
            )

    updated = document.copy()

    updated["status"] = new_status

    updated["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    if new_status == "failed":
        updated["error"] = error.strip()
    else:
        updated.pop("error", None)

    registry["documents"][document_id] = updated

    _atomic_write_json(
        Path(storage_root) / "registry.json",
        registry
    )

    return updated.copy()
