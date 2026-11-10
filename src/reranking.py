
import numpy as np
import torch

from sentence_transformers import CrossEncoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


# --------------------------------------------------
# 1. Load cross-encoder model
# --------------------------------------------------

def load_reranker_model(
    model_name=RERANKER_MODEL,
    device="cpu",
    max_length=512
):

    model = CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,

        # Return raw relevance logits,
        # rather than sigmoid-transformed values.
        activation_fn=torch.nn.Identity()
    )

    return model


# --------------------------------------------------
# 2. Prepare legal passage
# --------------------------------------------------

def prepare_passage(result):

    parts = []

    if result.get("document"):
        parts.append(
            f"Document: {result['document']}"
        )

    if result.get("chapter_title"):
        parts.append(
            f"Chapter: {result['chapter_title']}"
        )

    if result.get("section_title"):
        parts.append(
            f"Section: {result['section_title']}"
        )

    if result.get("article_title"):
        parts.append(
            f"Article title: {result['article_title']}"
        )

    if result.get("clause_reference"):
        parts.append(
            f"Legal provision: {result['clause_reference']}"
        )

    parts.append(
        f"Legal text: {result.get('text', '')}"
    )

    return "\n".join(parts)


# --------------------------------------------------
# 3. Rerank retrieved passages
# --------------------------------------------------

def rerank_results(
    question,
    candidates,
    model,
    top_k=10,
    batch_size=8
):

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not candidates:
        return []

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    # Construct question-passage pairs
    pairs = [
        (
            question,
            prepare_passage(candidate)
        )
        for candidate in candidates
    ]

    # Calculate cross-encoder relevance scores
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False
    )

    scores = np.asarray(scores)

    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]

    if scores.ndim != 1:
        raise ValueError(
            f"Expected one relevance score per passage, "
            f"received shape {scores.shape}"
        )

    if len(scores) != len(candidates):
        raise ValueError("Score count does not match candidates.")

    if not np.isfinite(scores).all():
        raise ValueError("Invalid cross-encoder scores.")

    reranked = []

    # Preserve existing citations and retrieval metadata
    for candidate, score in zip(candidates, scores):

        result = candidate.copy()

        result["pre_rerank_rank"] = candidate.get(
            "hybrid_rank",
            candidate.get("rank")
        )

        result["rerank_score"] = float(score)

        reranked.append(result)

    # Sort by descending relevance score
    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    # Assign new rankings
    reranked = reranked[:top_k]

    for rank, result in enumerate(reranked, start=1):

        result["rerank_rank"] = rank

    return reranked


# --------------------------------------------------
# 4. Compare rankings before and after reranking
# --------------------------------------------------

def compare_rankings(
    original_results,
    reranked_results
):

    comparison = []

    original_lookup = {
        item["chunk_id"]: item
        for item in original_results
    }

    for result in reranked_results:

        chunk_id = result["chunk_id"]

        original = original_lookup.get(chunk_id)

        comparison.append({

            "provision": result["clause_reference"],

            "original_rank": (
                original.get("hybrid_rank")
                if original else None
            ),

            "reranked_rank": result["rerank_rank"],

            "rerank_score": result["rerank_score"],

            "pages": result["page_numbers"]

        })

    return comparison
