import math

"""
Deterministic retrieval evaluation metrics.

These metrics operate on ranked chunk IDs and manually
labeled expected evidence IDs.

No embedding, retrieval, or LLM API calls occur here.
"""


def evaluate_retrieval_case(expected_chunk_ids, retrieved_chunk_ids, k=5):
    """
    Evaluate one answerable retrieval question.

    Hit@K: at least one expected chunk in top K.
    Recall@K: fraction of expected chunks in top K.
    Reciprocal rank: inverse rank of first relevant chunk.

    Unanswerable questions are evaluated separately.
    """

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    if not isinstance(expected_chunk_ids, (list, tuple, set)):
        raise TypeError("Expected chunk IDs must be a sequence.")

    if not isinstance(retrieved_chunk_ids, (list, tuple)):
        raise TypeError("Retrieved chunk IDs must be a ranked sequence.")

    expected = set(expected_chunk_ids)

    if not expected:
        raise ValueError(
            "Expected evidence cannot be empty for retrieval evaluation."
        )

    if any(not isinstance(x, str) or not x for x in expected):
        raise ValueError("Expected chunk IDs must be nonempty strings.")

    if any(
        not isinstance(x, str) or not x
        for x in retrieved_chunk_ids
    ):
        raise ValueError("Retrieved chunk IDs must be nonempty strings.")

    if len(retrieved_chunk_ids) != len(set(retrieved_chunk_ids)):
        raise ValueError("Duplicate retrieved chunk IDs are not allowed.")

    top_k = retrieved_chunk_ids[:k]

    relevant = expected.intersection(top_k)

    hit = int(bool(relevant))

    recall = len(relevant) / len(expected)

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit_at_k": hit,
        "recall_at_k": recall,
        "reciprocal_rank_at_k": reciprocal_rank,
        "relevant_retrieved": len(relevant),
        "relevant_expected": len(expected),
        "k": k,
    }


def summarize_retrieval_metrics(case_results):
    """
    Calculate macro-average metrics over answerable cases.
    """

    if not isinstance(case_results, list) or not case_results:
        raise ValueError("At least one evaluation result is required.")

    required = {
        "hit_at_k",
        "recall_at_k",
        "reciprocal_rank_at_k",
    }

    for result in case_results:
        if not isinstance(result, dict):
            raise TypeError("Each evaluation result must be a dictionary.")

        if not required.issubset(result):
            raise ValueError("Evaluation result is missing metric fields.")

        for field in required:
            value = result[field]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid metric value: {field}")

    count = len(case_results)

    return {
        "questions_evaluated": count,
        "hit_at_k": sum(
            x["hit_at_k"] for x in case_results
        ) / count,
        "recall_at_k": sum(
            x["recall_at_k"] for x in case_results
        ) / count,
        "mrr_at_k": sum(
            x["reciprocal_rank_at_k"] for x in case_results
        ) / count,
    }
