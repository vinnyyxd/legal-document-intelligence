
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.hybrid_retrieval import (
    hybrid_search,
    reciprocal_rank_fusion,
)


CHUNKS = [
    {
        "chunk_id": "doc_a_chunk_1",
        "document_id": "doc_a",
        "document": "Contract.pdf",
        "text": "Alpha contractual obligation.",
    },
    {
        "chunk_id": "doc_b_chunk_1",
        "document_id": "doc_b",
        "document": "Contract.pdf",
        "text": "Beta contractual obligation.",
    },
]


def dense_result(chunk_id, score=0.9):
    return {
        "chunk_id": chunk_id,
        "similarity_score": score,
    }


def sparse_result(chunk_id, score=5.0):
    return {
        "chunk_id": chunk_id,
        "bm25_score": score,
    }


def run_hybrid(dense, sparse, chunks=None):

    if chunks is None:
        chunks = CHUNKS

    faiss_index = SimpleNamespace(ntotal=len(chunks))
    bm25_index = SimpleNamespace(corpus_size=len(chunks))

    with (
        patch(
            "src.hybrid_retrieval.search_faiss",
            side_effect=dense
            if isinstance(dense, Exception)
            else None,
            return_value=None
            if isinstance(dense, Exception)
            else dense,
        ),
        patch(
            "src.hybrid_retrieval.search_bm25",
            side_effect=sparse
            if isinstance(sparse, Exception)
            else None,
            return_value=None
            if isinstance(sparse, Exception)
            else sparse,
        ),
    ):
        return hybrid_search(
            client=None,
            question="What obligations apply?",
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunks=chunks,
            top_k=5,
        )


def test_both_backends_succeed():

    results = run_hybrid(
        [
            dense_result("doc_a_chunk_1"),
            dense_result("doc_b_chunk_1"),
        ],
        [
            sparse_result("doc_a_chunk_1"),
            sparse_result("doc_b_chunk_1"),
        ],
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "doc_a_chunk_1"

    assert all(
        result["matched_by_both"]
        for result in results
    )

    assert results[0]["hybrid_rank"] == 1


def test_dense_temporary_failure():

    results = run_hybrid(
        RuntimeError("503 service unavailable"),
        [sparse_result("doc_b_chunk_1")],
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc_b_chunk_1"
    assert results[0]["retrieval_sources"] == ["bm25"]


def test_bm25_temporary_failure():

    results = run_hybrid(
        [dense_result("doc_a_chunk_1")],
        TimeoutError("BM25 timeout"),
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc_a_chunk_1"
    assert results[0]["retrieval_sources"] == ["dense"]


def test_both_backends_fail():

    with pytest.raises(
        RuntimeError,
        match="Both retrieval backends failed",
    ):
        run_hybrid(
            RuntimeError("503 unavailable"),
            TimeoutError("BM25 timeout"),
        )


def test_structural_mismatch():

    faiss_index = SimpleNamespace(ntotal=999)
    bm25_index = SimpleNamespace(corpus_size=2)

    with pytest.raises(ValueError, match="indexed vectors"):
        hybrid_search(
            client=None,
            question="Test question",
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunks=CHUNKS,
        )


def test_non_temporary_error_propagates():

    with pytest.raises(
        ValueError,
        match="Invalid vector dimension",
    ):
        run_hybrid(
            ValueError("Invalid vector dimension"),
            [sparse_result("doc_a_chunk_1")],
        )


def test_duplicate_rrf_results():

    results = reciprocal_rank_fusion(
        dense_results=[
            dense_result("doc_a_chunk_1"),
            dense_result("doc_a_chunk_1"),
        ],
        bm25_results=[],
        chunks=CHUNKS,
    )

    assert len(results) == 1
    assert results[0]["dense_rank"] == 1


def test_unknown_chunk_rejected():

    with pytest.raises(ValueError, match="Unknown chunk ID"):
        reciprocal_rank_fusion(
            dense_results=[
                dense_result("unknown_chunk")
            ],
            bm25_results=[],
            chunks=CHUNKS,
        )
