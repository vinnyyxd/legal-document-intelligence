
import faiss
import numpy as np

from pathlib import Path
from src.embeddings import embed_texts


# --------------------------------------------------
# 1. Build FAISS vector index
# --------------------------------------------------

def build_faiss_index(vectors):

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    if vectors.ndim != 2:
        raise ValueError("Vectors must be a 2D matrix.")

    if vectors.shape[0] == 0:
        raise ValueError("No vectors supplied.")

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors contain invalid values.")

    # Check whether vectors are normalized
    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "Vectors must be normalized before indexing."
        )

    dimension = vectors.shape[1]

    # Create exact inner-product index
    index = faiss.IndexFlatIP(dimension)

    # FAISS expects contiguous float32 arrays
    vectors = np.ascontiguousarray(vectors)

    # Add document vectors to index
    index.add(vectors)

    return index


# --------------------------------------------------
# 2. Save FAISS index
# --------------------------------------------------

def save_faiss_index(index, index_path):

    index_path = Path(index_path)

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    print("FAISS index saved:", index_path)


# --------------------------------------------------
# 3. Load FAISS index
# --------------------------------------------------

def load_faiss_index(index_path):

    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    index = faiss.read_index(
        str(index_path)
    )

    return index


# --------------------------------------------------
# 4. Semantic search
# --------------------------------------------------

def search_faiss(
    client,
    question,
    index,
    chunks,
    top_k=5
):

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if len(chunks) != index.ntotal:
        raise ValueError(
            "Number of chunks does not match indexed vectors."
        )

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    # Generate query embedding
    query_vector = embed_texts(
        client=client,
        texts=[question],
        task_type="RETRIEVAL_QUERY"
    )

    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32
    )

    if query_vector.shape[1] != index.d:
        raise ValueError(
            "Query and index dimensions do not match."
        )

    # Search nearest vectors
    k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_vector,
        k
    )

    results = []

    for rank, (score, position) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if position < 0:
            continue

        chunk = chunks[int(position)]

        results.append({

            "rank": rank,

            "similarity_score": float(score),

            "chunk_id": chunk["chunk_id"],
            **({"document_id": chunk["document_id"]} if "document_id" in chunk else {}),

            "document": chunk["document"],

            "article": chunk["article"],

            "paragraph": chunk["paragraph"],

            "clause_reference": chunk["clause_reference"],

            "page_numbers": chunk["page_numbers"],

            "text": chunk["text"]

        })

    return results
