import json
from pathlib import Path

import numpy as np

from src.document_registry import (
    DEFAULT_STORAGE_ROOT,
    get_document,
)
from src.embedding_provenance import (
    validate_embedding_manifest,
)
from src.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)
from src.retrieval import load_faiss_index


def load_ready_document(
    document_id,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    """Load and validate one persisted, ready legal document."""

    document = get_document(
        document_id,
        storage_root=storage_root,
    )

    if document["status"] != "ready":
        raise ValueError(
            "Only ready documents can be loaded for retrieval."
        )

    document_id = document["document_id"]
    directory = Path(storage_root) / document_id

    required = {
        "chunks": directory / "chunks.json",
        "vectors": directory / "embeddings.npy",
        "index": directory / "faiss.index",
        "metadata": directory / "embedding_metadata.json",
        "provenance": directory / "embedding_provenance.json",
    }

    for name, artifact in required.items():
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Missing {name} artifact for document {document_id}."
            )

    chunks = json.loads(
        required["chunks"].read_text(encoding="utf-8")
    )

    metadata = json.loads(
        required["metadata"].read_text(encoding="utf-8")
    )

    provenance = json.loads(
        required["provenance"].read_text(encoding="utf-8")
    )

    vectors = np.load(
        required["vectors"],
        allow_pickle=False,
    )

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Invalid or empty document chunks.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    if not all(
        chunk.get("document_id") == document_id
        and chunk.get("document") == document["filename"]
        for chunk in chunks
    ):
        raise ValueError("Document chunk identity mismatch.")

    if vectors.shape != (
        len(chunks),
        EMBEDDING_DIMENSION,
    ):
        raise ValueError("Stored embedding shape mismatch.")

    if vectors.dtype != np.float32:
        raise ValueError("Stored embeddings must be float32.")

    if not np.isfinite(vectors).all():
        raise ValueError("Stored embeddings contain invalid values.")

    if not np.allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError("Stored embeddings are not normalized.")

    if not isinstance(metadata, dict):
        raise ValueError("Invalid embedding metadata.")

    expected_fields = {
        "document_id": document_id,
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSION,
        "normalized": True,
        "task_type": "RETRIEVAL_DOCUMENT",
        "total_chunks": len(chunks),
    }

    for key, expected in expected_fields.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Embedding metadata mismatch: {key}"
            )

    expected_rows = [
        {
            "row_index": index,
            "chunk_id": chunk["chunk_id"],
            "document_id": document_id,
        }
        for index, chunk in enumerate(chunks)
    ]

    if metadata.get("rows") != expected_rows:
        raise ValueError("Embedding row alignment mismatch.")

    validate_embedding_manifest(
        provenance,
        chunks,
        vectors,
        expected_model=EMBEDDING_MODEL,
        expected_task_type="RETRIEVAL_DOCUMENT",
    )

    index = load_faiss_index(required["index"])

    if (
        index.ntotal != len(chunks)
        or index.d != EMBEDDING_DIMENSION
    ):
        raise ValueError("Persisted FAISS index shape mismatch.")

    # Verify the actual stored index contents, not just its size.
    for position, vector in enumerate(vectors):
        indexed_vector = index.reconstruct(position)

        if not np.array_equal(indexed_vector, vector):
            raise ValueError(
                "Persisted FAISS index vector mismatch."
            )

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "chunks": chunks,
        "vectors": vectors,
        "index": index,
        "metadata": metadata,
        "provenance": provenance,
    }


def build_multi_document_search(
    document_ids,
    storage_root=DEFAULT_STORAGE_ROOT,
):
    """Build temporary search indexes across selected ready documents."""

    from src.retrieval import build_faiss_index
    from src.bm25_retrieval import build_bm25_index

    if not isinstance(document_ids, (list, tuple)):
        raise ValueError("document_ids must be a list or tuple.")

    if not document_ids:
        raise ValueError("At least one document must be selected.")

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document IDs are not allowed.")

    all_chunks = []
    all_vectors = []
    loaded_documents = []

    seen_chunk_ids = set()

    for document_id in document_ids:

        document = load_ready_document(
            document_id,
            storage_root=storage_root,
        )

        for chunk in document["chunks"]:

            chunk_id = chunk["chunk_id"]

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID across documents: {chunk_id}"
                )

            seen_chunk_ids.add(chunk_id)

        all_chunks.extend(document["chunks"])
        all_vectors.append(document["vectors"])

        loaded_documents.append({
            "document_id": document["document_id"],
            "filename": document["filename"],
            "chunk_count": len(document["chunks"]),
        })

    combined_vectors = np.ascontiguousarray(
        np.concatenate(all_vectors, axis=0),
        dtype=np.float32,
    )

    combined_index = build_faiss_index(combined_vectors)

    combined_bm25 = build_bm25_index(all_chunks)

    if combined_index.ntotal != len(all_chunks):
        raise ValueError(
            "Combined FAISS index and chunk count mismatch."
        )

    return {
        "documents": loaded_documents,
        "chunks": all_chunks,
        "faiss_index": combined_index,
        "bm25_index": combined_bm25,
        "total_chunks": len(all_chunks),
    }


def search_multiple_documents(
    client,
    question,
    search_context,
    dense_k=20,
    sparse_k=20,
    top_k=10,
    rrf_k=60,
):
    """Search selected documents using the existing hybrid RAG engine."""

    from src.hybrid_retrieval import hybrid_search

    if not isinstance(search_context, dict):
        raise ValueError("Invalid multi-document search context.")

    required = {
        "chunks",
        "faiss_index",
        "bm25_index",
    }

    if not required.issubset(search_context):
        raise ValueError(
            "Multi-document search context is incomplete."
        )

    results = hybrid_search(
        client=client,
        question=question,
        faiss_index=search_context["faiss_index"],
        bm25_index=search_context["bm25_index"],
        chunks=search_context["chunks"],
        dense_k=dense_k,
        sparse_k=sparse_k,
        top_k=top_k,
        rrf_k=rrf_k,
    )

    return results


def answer_multiple_documents(
    client,
    question,
    document_ids,
    reranker_model,
    storage_root=DEFAULT_STORAGE_ROOT,
    dense_k=20,
    sparse_k=20,
    retrieval_k=20,
    rerank_k=10,
):
    """Answer a legal question across selected ready documents.

    Reuses the existing hybrid retrieval, cross-encoder
    reranking, grounded generation, and citation verification.
    """

    from src.reranking import rerank_results
    from src.generation import generate_grounded_answer
    from src.verification import verify_generated_answer
    from src.legal_reasoning import (
        analyze_legal_question,
        analyze_evidence_coverage,
        build_legal_reasoning_report,
    )

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if reranker_model is None:
        raise ValueError("A reranker model is required.")

    if retrieval_k < 1 or rerank_k < 1:
        raise ValueError("Retrieval limits must be positive.")

    question_analysis = analyze_legal_question(
        question,
        document_count=len(document_ids),
    )

    search_context = build_multi_document_search(
        document_ids,
        storage_root=storage_root,
    )

    retrieved = search_multiple_documents(
        client=client,
        question=question,
        search_context=search_context,
        dense_k=dense_k,
        sparse_k=sparse_k,
        top_k=retrieval_k,
    )

    if not retrieved:
        return {
            "evidence_coverage": None,
            "question_analysis": question_analysis,
            "question": question,
            "status": "insufficient_evidence",
            "answer": (
                "The documents do not contain enough verified "
                "evidence to answer this question."
            ),
            "reason": "No relevant evidence retrieved.",
            "documents": search_context["documents"],
            "retrieved_results": [],
            "reranked_results": [],
            "generation": None,
            "verification": None,
        }

    reranked = rerank_results(
        question=question,
        candidates=retrieved,
        model=reranker_model,
        top_k=rerank_k,
    )

    if not reranked:
        return {
            "evidence_coverage": None,
            "question_analysis": question_analysis,
            "question": question,
            "status": "insufficient_evidence",
            "answer": (
                "The documents do not contain enough verified "
                "evidence to answer this question."
            ),
            "reason": "No evidence remained after reranking.",
            "documents": search_context["documents"],
            "retrieved_results": retrieved,
            "reranked_results": [],
            "generation": None,
            "verification": None,
        }

    generated = generate_grounded_answer(
        client=client,
        question=question,
        retrieved_results=reranked,
    )

    evidence_coverage = analyze_evidence_coverage(
        question_analysis=question_analysis,
        retrieved_results=reranked,
        generated_result=generated,
    )

    verification = verify_generated_answer(
        client=client,
        question=question,
        generated_result=generated,
        retrieved_results=reranked,
        all_chunks=search_context["chunks"],
    )

    allowed_statuses = {
        "verified",
        "insufficient_evidence",
        "conflicting_evidence",
    }

    final_status = verification.get("status")

    if final_status not in allowed_statuses:
        raise ValueError(
            "Verifier returned an invalid final status."
        )

    final_answer = verification.get("answer")

    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError(
            "Verifier returned an invalid final answer."
        )

    reasoning_report = build_legal_reasoning_report(
        question_analysis=question_analysis,
        evidence_coverage=evidence_coverage,
        generated_result=generated,
        verification=verification,
    )

    return {
        "reasoning_report": reasoning_report,
        "evidence_coverage": evidence_coverage,
            "question_analysis": question_analysis,
        "question": question,
        "status": final_status,
        "answer": final_answer,
        "confidence": verification.get("confidence"),
        "documents": search_context["documents"],
        "retrieved_results": retrieved,
        "reranked_results": reranked,
        "generation": generated,
        "verification": verification,
    }
