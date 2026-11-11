
import json
import re

from google.genai import types
from src.gemini_utils import generate_with_retry


# --------------------------------------------------
# Configuration
# --------------------------------------------------

GENERATION_MODEL = "gemini-3.5-flash-lite"


# --------------------------------------------------
# 1. Build evidence context
# --------------------------------------------------

def build_evidence_context(results):

    if not results:
        raise ValueError("No retrieved evidence supplied.")

    evidence_map = {}
    evidence_blocks = []

    for index, result in enumerate(results, start=1):

        evidence_id = f"E{index}"

        document = result.get("document", "Unknown document")

        clause = result.get("clause_reference")

        pages = result.get("page_numbers", [])

        text = result.get("text", "").strip()

        if not text:
            continue

        if not pages:
            raise ValueError(
                f"Missing page metadata for {evidence_id}"
            )

        if not clause:
            raise ValueError(
                f"Missing legal reference for {evidence_id}"
            )

        evidence_map[evidence_id] = {
            "document": document,
            "clause_reference": clause,
            "page_numbers": pages,
            "chunk_id": result["chunk_id"],
        **({"document_id": result["document_id"]} if "document_id" in result else {})
        }

        block = f"""
EVIDENCE ID: {evidence_id}
Document: {document}
Legal provision: {clause}
PDF pages: {pages}

Legal text:
{text}
"""

        evidence_blocks.append(block)

    if not evidence_blocks:
        raise ValueError("No usable evidence passages found.")

    return "\n".join(evidence_blocks), evidence_map


# --------------------------------------------------
# 2. Validate generated JSON
# --------------------------------------------------

def validate_generated_output(payload, evidence_map):

    if not isinstance(payload, dict):
        raise ValueError("Output must be a JSON object.")

    status = payload.get("status")

    if status not in ["answered", "insufficient_evidence"]:
        raise ValueError("Invalid answer status.")

    claims = payload.get("claims")

    if not isinstance(claims, list):
        raise ValueError("Claims must be a list.")

    if status == "answered" and not claims:
        raise ValueError(
            "Answered status requires at least one claim."
        )

    for claim in claims:

        if not isinstance(claim, dict):
            raise ValueError("Each claim must be an object.")

        if not isinstance(claim.get("text"), str):
            raise ValueError("Claim text must be a string.")

        if not claim["text"].strip():
            raise ValueError("Empty claim detected.")

        evidence_ids = claim.get("evidence_ids")

        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError(
                "Every factual claim needs evidence IDs."
            )

        for evidence_id in evidence_ids:

            if evidence_id not in evidence_map:
                raise ValueError(
                    f"Unknown evidence ID: {evidence_id}"
                )

    return True


# --------------------------------------------------
# 3. Format actual citations
# --------------------------------------------------

def format_cited_answer(claims, evidence_map):

    formatted_claims = []

    for claim in claims:

        citations = []

        for evidence_id in claim["evidence_ids"]:

            source = evidence_map[evidence_id]

            document = source["document"]

            reference = source["clause_reference"]

            pages = source["page_numbers"]

            if len(pages) == 1:
                page_label = f"PDF p. {pages[0]}"
            else:
                page_label = (
                    "PDF pp. "
                    + ", ".join(str(p) for p in pages)
                )

            citation = (
                f"[{document}, {reference}, {page_label}]"
            )

            if citation not in citations:
                citations.append(citation)

        formatted_claims.append(
            f"- {claim['text']} {' '.join(citations)}"
        )

    return "\n\n".join(formatted_claims)


# --------------------------------------------------
# 4. Generate grounded legal answer
# --------------------------------------------------

def generate_grounded_answer(
    client,
    question,
    retrieved_results,
    model=GENERATION_MODEL,
    max_output_tokens=1800
):

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Prepare retrieved legal evidence
    evidence_context, evidence_map = build_evidence_context(
        retrieved_results
    )

    system_prompt = """
You are a legal document question-answering assistant.

Your responsibility is to answer questions using ONLY
the supplied legal document passages.

STRICT RULES:

1. Do not use external knowledge or pretrained knowledge
   to introduce legal facts.

2. Do not invent facts, clauses, article numbers,
   document names or page numbers.

3. Every factual or legal claim must be supported by
   one or more supplied evidence IDs.

4. Separate complex answers into clear, atomic claims.

5. Preserve important legal qualifications such as
   "whichever is higher", percentages, monetary limits,
   exceptions and conditions.

6. When combining multiple passages, explain how
   they contribute to the answer.

7. Distinguish general rules from specific exceptions.

8. Do not interpret a provision more broadly than
   its supplied text supports.

9. If the retrieved evidence cannot answer the question,
   set status to "insufficient_evidence" and provide
   an explanation. Do not guess.

10. Evidence IDs are internal references. Do not
    generate your own citations or page numbers.

OUTPUT FORMAT:

Return ONLY valid JSON with this structure:

{
    "status": "answered",
    "claims": [
        {
            "text": "A supported factual statement.",
            "evidence_ids": ["E1"]
        }
    ],
    "insufficient_evidence_reason": ""
}

For insufficient evidence:

{
    "status": "insufficient_evidence",
    "claims": [],
    "insufficient_evidence_reason": "Explanation."
}

Use only evidence IDs that actually appear
in the supplied evidence.
"""

    prompt = f"""
{system_prompt}

USER QUESTION:
{question}

RETRIEVED LEGAL EVIDENCE:
{evidence_context}

Generate the evidence-grounded JSON answer now.
"""

    # Generate structured JSON response
    response = generate_with_retry(
        client=client,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=max_output_tokens
        ),
        fallback_models=["gemini-3.8-flash"],
        max_attempts=2,
        base_delay=15
    )

    raw_text = response.text

    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    # Remove optional Markdown fences
    raw_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_text.strip(),
        flags=re.IGNORECASE
    )

    # Parse JSON
    try:
        payload = json.loads(raw_text)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Could not parse Gemini JSON response: {exc}"
        ) from exc

    # Validate structure and evidence references
    validate_generated_output(
        payload,
        evidence_map
    )

    if payload["status"] == "answered":

        answer = format_cited_answer(
            payload["claims"],
            evidence_map
        )

    else:

        answer = (
            "The retrieved documents do not contain "
            "enough evidence to answer this question.\n\n"
            + payload.get(
                "insufficient_evidence_reason", ""
            )
        )

    return {
        "question": question,
        "status": payload["status"],
        "claims": payload["claims"],
        "answer": answer,
        "evidence_map": evidence_map,
        "model": model,
        "raw_response": raw_text
    }
