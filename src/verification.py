
import json
import re

from google.genai import types
from src.gemini_utils import generate_with_retry


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VERIFIER_MODEL = "gemini-3.5-flash-lite"


# --------------------------------------------------
# 1. Validate citation integrity using Python
# --------------------------------------------------

def validate_citation_integrity(
    generated_result,
    retrieved_results
):

    errors = []

    evidence_map = generated_result.get(
        "evidence_map", {}
    )

    claims = generated_result.get("claims", [])

    retrieved_lookup = {
        result["chunk_id"]: result
        for result in retrieved_results
    }

    if generated_result.get("status") == "answered" and not claims:
        errors.append("Answered response has no claims.")

    for claim_index, claim in enumerate(claims, start=1):

        evidence_ids = claim.get("evidence_ids")

        if not isinstance(evidence_ids, list) or not evidence_ids:
            errors.append(
                f"Claim {claim_index} has no evidence IDs."
            )
            continue

        for evidence_id in evidence_ids:

            source = evidence_map.get(evidence_id)

            if source is None:
                errors.append(
                    f"Claim {claim_index}: unknown evidence ID "
                    f"{evidence_id}."
                )
                continue

            chunk_id = source.get("chunk_id")

            retrieved_chunk = retrieved_lookup.get(chunk_id)

            if retrieved_chunk is None:
                errors.append(
                    f"{evidence_id} does not reference a "
                    "retrieved passage."
                )
                continue

            # Verify document identity
            if (
                source["document"]
                != retrieved_chunk["document"]
            ):
                errors.append(
                    f"Document mismatch for {evidence_id}."
                )

            # Verify legal clause identity
            if (
                source["clause_reference"]
                != retrieved_chunk["clause_reference"]
            ):
                errors.append(
                    f"Clause mismatch for {evidence_id}."
                )

            # Verify exact page mapping
            if (
                source["page_numbers"]
                != retrieved_chunk["page_numbers"]
            ):
                errors.append(
                    f"Page mismatch for {evidence_id}."
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "claims_checked": len(claims)
    }


# --------------------------------------------------
# 2. Build verification context
# --------------------------------------------------

def build_verification_context(
    generated_result,
    retrieved_results
):

    evidence_map = generated_result["evidence_map"]

    retrieved_lookup = {
        result["chunk_id"]: result
        for result in retrieved_results
    }

    evidence_blocks = []

    for evidence_id, source in evidence_map.items():

        chunk = retrieved_lookup.get(
            source["chunk_id"]
        )

        if chunk is None:
            continue

        block = f"""
EVIDENCE ID: {evidence_id}

Document: {source['document']}
Legal provision: {source['clause_reference']}
PDF pages: {source['page_numbers']}

Text:
{chunk['text']}
"""

        evidence_blocks.append(block)

    return "\n".join(evidence_blocks)


# --------------------------------------------------
# 3. Verify claims and question sufficiency
# --------------------------------------------------



# --------------------------------------------------
# Confidence classification
# --------------------------------------------------

def classify_confidence(
    question_status,
    claim_checks,
    citation_integrity_valid,
    generator_abstained=False
):

    statuses = [
        item["status"] for item in claim_checks
    ]

    if not citation_integrity_valid:
        return "LOW"

    if question_status == "conflicting":
        return "LOW"

    if "conflicting" in statuses:
        return "LOW"

    if generator_abstained or question_status == "insufficient":
        return "INSUFFICIENT EVIDENCE"

    if not statuses:
        return "INSUFFICIENT EVIDENCE"

    if all(status == "supported" for status in statuses):
        return "HIGH"

    if all(
        status in {"supported", "partially_supported"}
        for status in statuses
    ):
        return "MEDIUM"

    return "LOW"



# --------------------------------------------------
# Format citations for conflicting sources
# --------------------------------------------------

def format_source_citations(claims, evidence_map):

    citations = []

    for claim in claims:

        for evidence_id in claim.get("evidence_ids", []):

            evidence = evidence_map.get(evidence_id)

            if not evidence:
                continue

            pages = evidence["page_numbers"]

            page_label = (
                f"PDF p. {pages[0]}"
                if len(pages) == 1
                else "PDF pp. " + ", ".join(map(str, pages))
            )

            citation = (
                f"[{evidence['document']}, "
                f"{evidence['clause_reference']}, "
                f"{page_label}]"
            )

            if citation not in citations:
                citations.append(citation)

    return citations


def verify_generated_answer(
    client,
    question,
    generated_result,
    retrieved_results,
    all_chunks=None,
    model=VERIFIER_MODEL
):

    # First perform deterministic citation checks
    integrity = validate_citation_integrity(
        generated_result,
        retrieved_results
    )

    # Identify the documents in the indexed corpus
    corpus = (
        all_chunks
        if all_chunks is not None
        else retrieved_results
    )

    documents_searched = sorted({
        chunk["document"]
        for chunk in corpus
    })

    def abstain(status, reason):

        return {
            "question": question,
            "status": status,
            "answer": (
                "The documents do not contain enough "
                "verified evidence to answer this question.\n\n"
                + reason
            ),
            "documents_searched": documents_searched,
            "citation_integrity": integrity,
            "claim_checks": [],
            "model": model
        }

    # Never continue with invalid citations
    if not integrity["valid"]:

        return abstain(
            "insufficient_evidence",
            "Citation integrity verification failed: "
            + "; ".join(integrity["errors"])
        )

        # Record generator abstention.
    # Still run independent question sufficiency verification.
    generator_abstained = (
        generated_result.get("status") != "answered"
    )

    evidence_context = build_verification_context(
        generated_result,
        retrieved_results
    )

    claims_context = json.dumps(
        [
            {
                "claim_index": index,
                "claim": claim["text"],
                "cited_evidence_ids": claim["evidence_ids"]
            }
            for index, claim in enumerate(
                generated_result["claims"],
                start=1
            )
        ],
        ensure_ascii=False,
        indent=2
    )

    verifier_prompt = f"""
You are an independent legal evidence verifier.

You must use ONLY the supplied retrieved document passages.

USER QUESTION:
{question}

GENERATED CLAIMS:
{claims_context}

RETRIEVED EVIDENCE:
{evidence_context}

YOUR TASK:

A. QUESTION SUFFICIENCY

Determine whether the retrieved evidence collectively
contains enough information to answer ALL material
parts of the user question.

Return exactly one status:

- sufficient
- insufficient
- conflicting

Do not confuse topical relevance with sufficient evidence.

If only part of a multi-part question can be answered,
choose insufficient.

If sources give materially inconsistent answers
and the conflict cannot be resolved from the supplied
documents, choose conflicting.

B. CLAIM VERIFICATION

Evaluate every generated claim separately.

IMPORTANT:
- Evaluate each claim using only its cited evidence IDs.
- Check monetary values, percentages and legal conditions.
- Check whether the evidence actually entails the claim.
- Do not accept a merely related passage as support.
- Identify unsupported extensions or omitted qualifications.

Each claim must receive exactly one status:

- supported
- partially_supported
- unsupported
- conflicting

C. OUTPUT

Return ONLY valid JSON:

{{
    "question_status": "sufficient",
    "question_reason": "Explanation.",
    "missing_aspects": [],
    "claim_checks": [
        {{
            "claim_index": 1,
            "status": "supported",
            "reason": "Explanation."
        }}
    ]
}}

Include one claim check for every supplied claim.
Do not invent claim indices.
"""

    response = generate_with_retry(
        client=client,
        model=model,
        contents=verifier_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            max_output_tokens=1800
        ),
        fallback_models=["gemini-3.8-flash"],
        max_attempts=2,
        base_delay=15
    )

    raw_text = response.text

    if not raw_text:
        raise ValueError("Verifier returned an empty response.")

    raw_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_text.strip(),
        flags=re.IGNORECASE
    )

    payload = json.loads(raw_text)

    # Validate question assessment
    question_status = payload.get("question_status")

    allowed_question_statuses = {
        "sufficient",
        "insufficient",
        "conflicting"
    }

    if question_status not in allowed_question_statuses:
        raise ValueError("Invalid question verification status.")

    # Validate claim assessments
    claim_checks = payload.get("claim_checks")

    if not isinstance(claim_checks, list):
        raise ValueError("Claim checks must be a list.")

    expected_count = len(generated_result["claims"])

    if len(claim_checks) != expected_count:
        raise ValueError(
            "Verifier did not assess every generated claim."
        )

    expected_indices = set(range(1, expected_count + 1))

    returned_indices = {
        item.get("claim_index")
        for item in claim_checks
    }

    if returned_indices != expected_indices:
        raise ValueError("Invalid or duplicate claim indices.")

    allowed_claim_statuses = {
        "supported",
        "partially_supported",
        "unsupported",
        "conflicting"
    }

    for item in claim_checks:

        if item.get("status") not in allowed_claim_statuses:
            raise ValueError("Invalid claim verification status.")

    # --------------------------------------------------
    # 4. Finalize answer or abstain
    # --------------------------------------------------

    any_conflict = (
        question_status == "conflicting"
        or any(
            item["status"] == "conflicting"
            for item in claim_checks
        )
    )

    all_claims_supported = all(
        item["status"] == "supported"
        for item in claim_checks
    )

    if any_conflict:

        final_status = "conflicting_evidence"

    elif (
        generator_abstained
        or question_status != "sufficient"
        or not all_claims_supported
        or not generated_result.get("claims")
    ):

        final_status = "insufficient_evidence"

    else:

        final_status = "verified"

    if final_status == "verified":

        final_answer = generated_result["answer"]

    else:

        reasons = [
            payload.get("question_reason", "")
        ]

        reasons.extend(
            payload.get("missing_aspects", [])
        )

        reasons.extend(
            item["reason"]
            for item in claim_checks
            if item["status"] != "supported"
        )

        reason_text = "; ".join(
            reason for reason in reasons if reason
        ).strip()

        if final_status == "conflicting_evidence":

            source_citations = format_source_citations(
                generated_result["claims"],
                generated_result["evidence_map"]
            )

            final_answer = (
                "The retrieved documents contain conflicting "
                "evidence. A definitive answer cannot be given."
            )

            if source_citations:

                final_answer += (
                    "\\n\\nConflicting sources: "
                    + " ".join(source_citations)
                )

            if reason_text:

                final_answer += (
                    "\\n\\n" + reason_text
                )

        else:

            standard_message = (
                "The documents do not contain enough "
                "verified evidence to answer this question."
            )

            duplicate_phrases = (
                "the documents do not contain",
                "the retrieved documents do not contain",
                "the retrieved passages do not contain",
                "insufficient evidence"
            )

            is_duplicate = any(
                reason_text.lower().startswith(phrase)
                for phrase in duplicate_phrases
            )

            if reason_text and not is_duplicate:

                final_answer = (
                    standard_message
                    + "\n\nReason: "
                    + reason_text
                )

            else:

                final_answer = standard_message


    confidence = classify_confidence(
        question_status=question_status,
        claim_checks=claim_checks,
        citation_integrity_valid=integrity["valid"],
        generator_abstained=generator_abstained
    )

    return {
        "question": question,
        "status": final_status,
        "confidence": confidence,
        "answer": final_answer,
        "documents_searched": documents_searched,
        "citation_integrity": integrity,
        "question_assessment": {
            "status": question_status,
            "reason": payload.get("question_reason", ""),
            "missing_aspects": payload.get(
                "missing_aspects", []
            )
        },
        "claim_checks": claim_checks,
        "model": model,
        "raw_verifier_response": raw_text
    }
