
import re
import unicodedata
import numpy as np

from rank_bm25 import BM25Okapi


# --------------------------------------------------
# 1. Legal reference patterns
# --------------------------------------------------

ARTICLE_PATTERN = re.compile(
    r"\barticle\s+(\d+[a-z]?)",
    re.IGNORECASE
)

ARTICLE_CLAUSE_PATTERN = re.compile(
    r"\barticle\s+(\d+[a-z]?)\s*"
    r"\(\s*(\d+)\s*\)"
    r"(?:\s*\(\s*([a-z])\s*\))?",
    re.IGNORECASE
)

CLAUSE_ONLY_PATTERN = re.compile(
    r"\b(\d+)\s*\(\s*(\d+)\s*\)"
    r"(?:\s*\(\s*([a-z])\s*\))?"
)

TOKEN_PATTERN = re.compile(
    r"[a-z]+|\d+(?:[.,]\d+)?"
)


# --------------------------------------------------
# 2. Legal-aware tokenizer
# --------------------------------------------------

def tokenize_legal_text(text):

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    ).lower()

    # Normalize common legal financial notation
    text = text.replace("€", " eur ")
    text = text.replace("%", " percent ")

    # Extract normal words and numbers
    tokens = TOKEN_PATTERN.findall(text)

    # Add special article identifiers
    special_tokens = []

    for match in ARTICLE_PATTERN.finditer(text):

        article = match.group(1)

        special_tokens.append(
            f"article_{article}"
        )

    for match in ARTICLE_CLAUSE_PATTERN.finditer(text):

        article = match.group(1)
        paragraph = match.group(2)
        subclause = match.group(3)

        special_tokens.append(
            f"article_{article}_{paragraph}"
        )

        special_tokens.append(
            f"clause_{article}_{paragraph}"
        )

        if subclause:

            special_tokens.append(
                f"article_{article}_{paragraph}_{subclause}"
            )

            special_tokens.append(
                f"clause_{article}_{paragraph}_{subclause}"
            )

    # Also support references written as 83(4)
    for match in CLAUSE_ONLY_PATTERN.finditer(text):

        paragraph = match.group(1)
        subclause = match.group(2)
        letter = match.group(3)

        special_tokens.append(
            f"clause_{paragraph}_{subclause}"
        )

        if letter:

            special_tokens.append(
                f"clause_{paragraph}_{subclause}_{letter}"
            )

    # Avoid duplicate special identifiers
    seen = set()

    for token in special_tokens:

        if token not in seen:
            tokens.append(token)
            seen.add(token)

    return tokens


# --------------------------------------------------
# 3. Prepare searchable document text
# --------------------------------------------------

def prepare_document_text(chunk):

    fields = [

        chunk.get("document", ""),

        chunk.get("chapter_title", ""),

        chunk.get("section_title", ""),

        chunk.get("article_title", ""),

        chunk.get("clause_reference", ""),

        chunk.get("parent_clause_reference", ""),

        chunk.get("text", "")
    ]

    return " ".join(
        str(field)
        for field in fields
        if field
    )


# --------------------------------------------------
# 4. Build BM25 index
# --------------------------------------------------

def build_bm25_index(chunks, k1=1.5, b=0.75):

    if not chunks:
        raise ValueError("No legal chunks supplied.")

    if k1 <= 0:
        raise ValueError("k1 must be positive.")

    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1.")

    tokenized_documents = []

    for chunk in chunks:

        document_text = prepare_document_text(chunk)

        tokens = tokenize_legal_text(
            document_text
        )

        tokenized_documents.append(tokens)

    bm25 = BM25Okapi(
        tokenized_documents,
        k1=k1,
        b=b
    )

    return bm25


# --------------------------------------------------
# 5. Search legal documents using BM25
# --------------------------------------------------

def search_bm25(
    question,
    bm25,
    chunks,
    top_k=5
):

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if len(chunks) != bm25.corpus_size:
        raise ValueError(
            "Chunk count and BM25 corpus size differ."
        )

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    query_tokens = tokenize_legal_text(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    if not np.isfinite(scores).all():
        raise ValueError("BM25 returned invalid scores.")

    # Highest relevance score first
    ranked_indices = np.argsort(
        -scores,
        kind="stable"
    )

    results = []

    for position in ranked_indices:

        score = float(scores[position])

        # Ignore documents with no positive lexical match
        if score <= 0:
            continue

        chunk = chunks[int(position)]

        results.append({

            "rank": len(results) + 1,

            "bm25_score": score,

            "chunk_id": chunk["chunk_id"],

            "document": chunk["document"],

            "article": chunk["article"],

            "paragraph": chunk["paragraph"],

            "subclause": chunk.get("subclause"),

            "clause_reference": chunk["clause_reference"],

            "page_numbers": chunk["page_numbers"],

            "text": chunk["text"]

        })

        if len(results) >= top_k:
            break

    return results
