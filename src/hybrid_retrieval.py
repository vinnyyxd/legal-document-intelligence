
from src.retrieval import search_faiss
from src.bm25_retrieval import search_bm25


# --------------------------------------------------
# 1. Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    chunks,
    rrf_k=60,
    top_k=10
):

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    # Map chunk IDs to original legal chunks
    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    fused_scores = {}

    # Process dense and sparse rankings
    retrieval_lists = [
        ("dense", dense_results),
        ("bm25", bm25_results)
    ]

    for source, results in retrieval_lists:

        seen = set()

        for rank, result in enumerate(results, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id not in chunk_lookup:
                raise ValueError(
                    f"Unknown chunk ID: {chunk_id}"
                )

            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            if chunk_id not in fused_scores:

                fused_scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_similarity": None,
                    "bm25_score": None,
                    "retrieval_sources": []
                }

            entry = fused_scores[chunk_id]

            # Calculate reciprocal rank contribution
            contribution = 1.0 / (rrf_k + rank)

            entry["rrf_score"] += contribution

            entry[f"{source}_rank"] = rank

            if source == "dense":
                entry["dense_similarity"] = result[
                    "similarity_score"
                ]

            else:
                entry["bm25_score"] = result[
                    "bm25_score"
                ]

            entry["retrieval_sources"].append(source)

    # Attach original legal metadata and text
    fused_results = []

    for chunk_id, entry in fused_scores.items():

        original_chunk = chunk_lookup[chunk_id]

        final_result = {
            **original_chunk,
            **entry
        }

        final_result["matched_by_both"] = (
            len(entry["retrieval_sources"]) == 2
        )

        fused_results.append(final_result)

    # Sort by descending RRF score
    fused_results.sort(
        key=lambda item: (
            -item["rrf_score"],
            item["chunk_id"]
        )
    )

    # Assign final hybrid ranks
    for rank, result in enumerate(
        fused_results[:top_k],
        start=1
    ):
        result["hybrid_rank"] = rank

    return fused_results[:top_k]


# --------------------------------------------------
# 2. Hybrid retrieval pipeline
# --------------------------------------------------

def hybrid_search(
    client,
    question,
    faiss_index,
    bm25_index,
    chunks,
    dense_k=20,
    sparse_k=20,
    top_k=10,
    rrf_k=60
):
    """
    Hybrid retrieval with basic backend recovery.

    Temporary backend failures may fall back to the
    remaining retrieval method.

    Structural and programming errors are not suppressed.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if dense_k < 1 or sparse_k < 1 or top_k < 1:
        raise ValueError("Retrieval limits must be positive.")

    # Structural consistency must be checked before
    # attempting either backend.
    if len(chunks) != faiss_index.ntotal:
        raise ValueError(
            "Number of chunks does not match indexed vectors."
        )

    if len(chunks) != bm25_index.corpus_size:
        raise ValueError(
            "Chunk count and BM25 corpus size differ."
        )

    def is_temporary_failure(exc):
        message = str(exc).lower()

        temporary_markers = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource_exhausted",
            "unavailable",
            "deadline",
            "timeout",
            "temporarily",
        )

        return any(
            marker in message
            for marker in temporary_markers
        )

    dense_results = []
    sparse_results = []
    failures = []

    try:
        dense_results = search_faiss(
            client=client,
            question=question,
            index=faiss_index,
            chunks=chunks,
            top_k=dense_k,
        )
    except Exception as exc:
        if not is_temporary_failure(exc):
            raise

        failures.append(("dense", exc))

    try:
        sparse_results = search_bm25(
            question=question,
            bm25=bm25_index,
            chunks=chunks,
            top_k=sparse_k,
        )
    except Exception as exc:
        if not is_temporary_failure(exc):
            raise

        failures.append(("bm25", exc))

    # Do not silently convert complete backend failure
    # into an ordinary no-evidence result.
    if len(failures) == 2:
        raise RuntimeError(
            "Both retrieval backends failed temporarily."
        ) from failures[-1][1]

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=sparse_results,
        chunks=chunks,
        rrf_k=rrf_k,
        top_k=top_k,
    )
