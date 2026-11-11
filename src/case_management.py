"""
Case management for Legal Document Intelligence.

A case groups 2–5 legal PDF documents.
Existing document IDs and indexes remain unchanged.
"""

import json
import os
import tempfile
import uuid

from datetime import datetime, timezone
from pathlib import Path

from src.document_registry import (
    DEFAULT_STORAGE_ROOT,
    get_document,
)


MIN_DOCUMENTS = 2
MAX_DOCUMENTS = 5


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _case_registry_path(storage_root):
    return Path(storage_root) / "cases.json"


def _save_cases(cases, storage_root):
    path = _case_registry_path(storage_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_path = tempfile.mkstemp(
        prefix=".cases_",
        suffix=".json",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(cases, file, indent=2)

        os.replace(temporary_path, path)

    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_cases(storage_root=DEFAULT_STORAGE_ROOT):
    path = _case_registry_path(storage_root)

    if not path.exists():
        return {"cases": {}}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Invalid case registry format.")

    if not isinstance(data.get("cases"), dict):
        raise ValueError("Invalid cases collection.")

    return data


def create_case(
    name,
    description="",
    storage_root=DEFAULT_STORAGE_ROOT,
):
    name = str(name).strip()
    description = str(description).strip()

    if not name:
        raise ValueError("Case name cannot be empty.")

    if len(name) > 150:
        raise ValueError("Case name exceeds 150 characters.")

    registry = load_cases(storage_root)

    case_id = str(uuid.uuid4())
    timestamp = _utc_now()

    case = {
        "case_id": case_id,
        "name": name,
        "description": description,
        "document_ids": [],
        "created_at": timestamp,
        "updated_at": timestamp,
    }

    registry["cases"][case_id] = case
    _save_cases(registry, storage_root)

    return case


def get_case(
    case_id,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    registry = load_cases(storage_root)
    case = registry["cases"].get(case_id)

    if case is None:
        raise KeyError(f"Case not found: {case_id}")

    return case


def list_cases(storage_root=DEFAULT_STORAGE_ROOT):
    registry = load_cases(storage_root)

    return sorted(
        registry["cases"].values(),
        key=lambda case: case["created_at"],
        reverse=True,
    )


def add_document_to_case(
    case_id,
    document_id,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    registry = load_cases(storage_root)

    case = registry["cases"].get(case_id)

    if case is None:
        raise KeyError(f"Case not found: {case_id}")

    # Confirm document exists in the existing registry.
    document = get_document(
        document_id,
        storage_root=storage_root,
    )

    if document is None:
        raise ValueError(
            f"Document not found: {document_id}"
        )

    document_ids = case["document_ids"]

    if document_id in document_ids:
        return case

    if len(document_ids) >= MAX_DOCUMENTS:
        raise ValueError(
            "A case cannot contain more than 5 documents."
        )

    document_ids.append(document_id)
    case["updated_at"] = _utc_now()

    _save_cases(registry, storage_root)

    return case


def remove_document_from_case(
    case_id,
    document_id,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    registry = load_cases(storage_root)

    case = registry["cases"].get(case_id)

    if case is None:
        raise KeyError(f"Case not found: {case_id}")

    if document_id not in case["document_ids"]:
        raise ValueError(
            "Document is not assigned to this case."
        )

    case["document_ids"].remove(document_id)
    case["updated_at"] = _utc_now()

    _save_cases(registry, storage_root)

    return case


def get_case_documents(
    case_id,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    case = get_case(case_id, storage_root)

    return [
        get_document(
            document_id,
            storage_root=storage_root,
        )
        for document_id in case["document_ids"]
    ]


def validate_case_for_query(
    case_id,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    case = get_case(case_id, storage_root)

    document_ids = case["document_ids"]

    if not MIN_DOCUMENTS <= len(document_ids) <= MAX_DOCUMENTS:
        raise ValueError(
            "A case must contain 2–5 documents before querying."
        )

    documents = get_case_documents(
        case_id,
        storage_root=storage_root,
    )

    for document in documents:
        if document is None:
            raise ValueError(
                "A case references a missing document."
            )

        if document.get("status") != "ready":
            raise ValueError(
                "All case documents must be indexed and ready."
            )

    return {
        "case_id": case_id,
        "document_ids": document_ids.copy(),
        "documents": documents,
        "valid": True,
    }
