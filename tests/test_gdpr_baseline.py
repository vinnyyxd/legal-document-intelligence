import hashlib
import json
from pathlib import Path

import faiss


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CHUNKS = 397
EXPECTED_PAGES = 88
EXPECTED_DIMENSIONS = 768

EXPECTED_FINGERPRINT = "e057dd0bec7cd4aa216cf65258bd9a73354e1c28007cce0bca7efcff8677009b"


def load_json(relative_path):
    path = ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def test_gdpr_chunk_baseline():

    chunks = load_json(
        "data/sample_documents/gdpr_chunks.json"
    )

    assert len(chunks) == EXPECTED_CHUNKS

    ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(set(ids)) == EXPECTED_CHUNKS

    canonical = json.dumps(
        chunks,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    actual = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    assert actual == EXPECTED_FINGERPRINT


def test_gdpr_extraction_baseline():

    pages = load_json(
        "data/sample_documents/gdpr_extracted.json"
    )

    assert isinstance(pages, list)
    assert len(pages) == EXPECTED_PAGES

    assert all(
        isinstance(page, dict)
        and "text" in page
        and "page" in page
        for page in pages
    )


def test_gdpr_faiss_baseline():

    path = (
        ROOT /
        "data/embeddings/gdpr_faiss.index"
    )

    index = faiss.read_index(str(path))

    assert index.ntotal == EXPECTED_CHUNKS
    assert index.d == EXPECTED_DIMENSIONS
