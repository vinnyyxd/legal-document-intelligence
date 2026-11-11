"""
Deterministic legal question analysis.

Phase 16: Advanced Legal Reasoning.

Question classification is advisory and does not
replace evidence-grounded generation or verification.
"""

import re


COMPARISON_PATTERNS = (
    r"\\bcompar(?:e|es|ed|ing|ison)\\b",
    r"\\bdiffer(?:ence|ences|ent|s)?\\b",
    r"\\bversus\\b",
    r"\\bvs\\.?\\b",
    r"\\bsimilarit(?:y|ies)\\b",
    r"\\bwhereas\\b",
)

CONFLICT_PATTERNS = (
    r"\\bconflict(?:s|ing)?\\b",
    r"\\bcontradict(?:s|ion|ions|ory)?\\b",
    r"\\binconsisten(?:t|cy|cies)\\b",
    r"\\bincompatib(?:le|ility)\\b",
    r"\\bdisagree(?:ment|ments)?\\b",
)

MULTIPART_PATTERNS = (
    r"\\band (?:how|why|when|whether|what|which)\\b",
    r"\\b(?:both|respectively)\\b",
    r"\\b(?:first|second|third)\\b",
    r"\\b(?:requirements|obligations|exceptions)\\b",
)


def _matches_any(text, patterns):
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def analyze_legal_question(question, document_count=None):
    """
    Return a deterministic, advisory analysis of a question.

    This does not assert that a legal conflict exists or
    that retrieved evidence is sufficient.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if document_count is not None:
        if (
            not isinstance(document_count, int)
            or isinstance(document_count, bool)
            or document_count < 1
        ):
            raise ValueError(
                "document_count must be a positive integer."
            )

    question = question.strip()

    comparison = _matches_any(
        question,
        COMPARISON_PATTERNS,
    )

    conflict = _matches_any(
        question,
        CONFLICT_PATTERNS,
    )

    multipart = _matches_any(
        question,
        MULTIPART_PATTERNS,
    )

    if conflict:
        question_type = "conflict_analysis"
    elif comparison:
        question_type = "comparison"
    elif multipart:
        question_type = "multi_part"
    else:
        question_type = "direct"

    return {
        "question": question,
        "question_type": question_type,
        "comparison_requested": comparison,
        "conflict_requested": conflict,
        "multi_part_requested": multipart,
        "multiple_documents_selected": (
            document_count is not None
            and document_count > 1
        ),
        "document_count": document_count,
        "analysis_method": "deterministic",
    }



def analyze_evidence_coverage(
    question_analysis,
    retrieved_results,
    generated_result,
):
    """
    Analyze document participation in generated claims.

    Uses structured evidence IDs rather than parsing
    human-readable legal citations.

    Coverage is advisory. It does not establish legal
    correctness, sufficiency, or verified claim support.
    """

    if not isinstance(question_analysis, dict):
        raise TypeError("question_analysis must be a dictionary.")

    if not isinstance(retrieved_results, list):
        raise TypeError("retrieved_results must be a list.")

    evidence_map = {}

    for index, result in enumerate(
        retrieved_results,
        start=1,
    ):
        evidence_id = f"E{index}"

        evidence_map[evidence_id] = {
            "document": result.get("document"),
            "document_id": result.get("document_id"),
            "chunk_id": result.get("chunk_id"),
            "page_numbers": result.get("page_numbers", []),
        }

    claims = (
        generated_result.get("claims", [])
        if isinstance(generated_result, dict)
        else []
    )

    cited_ids = set()

    for claim in claims:
        if not isinstance(claim, dict):
            continue

        for evidence_id in claim.get("evidence_ids", []):
            if (
                isinstance(evidence_id, str)
                and evidence_id in evidence_map
            ):
                cited_ids.add(evidence_id)

    def document_key(evidence):
        # Prefer stable document identity over filenames.
        document_id = evidence.get("document_id")

        if document_id:
            return f"id:{document_id}"

        return f"name:{evidence.get('document') or 'Unknown'}"

    retrieved_documents = {
        document_key(evidence)
        for evidence in evidence_map.values()
    }

    cited_documents = {
        document_key(evidence_map[evidence_id])
        for evidence_id in cited_ids
    }

    cited_evidence = [
        {
            "evidence_id": evidence_id,
            **evidence_map[evidence_id],
        }
        for evidence_id in sorted(
            cited_ids,
            key=lambda value: int(value[1:]),
        )
    ]

    comparison_requested = bool(
        question_analysis.get("comparison_requested")
        or question_analysis.get("conflict_requested")
    )

    multiple_documents_selected = bool(
        question_analysis.get("multiple_documents_selected")
    )

    cross_document_coverage = (
        len(cited_documents) >= 2
    )

    coverage_warning = None

    if (
        comparison_requested
        and multiple_documents_selected
        and not cross_document_coverage
    ):
        coverage_warning = (
            "The generated claims cite fewer than two "
            "distinct documents. Review whether the "
            "requested cross-document comparison is "
            "adequately supported."
        )

    return {
        "retrieved_document_count": len(retrieved_documents),
        "cited_document_count": len(cited_documents),
        "cited_evidence_count": len(cited_ids),
        "cited_documents": sorted(cited_documents),
        "cited_evidence": cited_evidence,
        "cross_document_coverage": cross_document_coverage,
        "coverage_warning": coverage_warning,
        "analysis_method": "deterministic",
        "verification_status": "not_assessed",
    }



def build_legal_reasoning_report(
    question_analysis,
    evidence_coverage,
    generated_result,
    verification,
):
    """
    Build an advisory reasoning report.

    The verifier's final status is authoritative.
    Generated claims are not independently verified
    by this reporting function.
    """

    if not isinstance(question_analysis, dict):
        raise TypeError("Invalid question analysis.")

    if not isinstance(evidence_coverage, dict):
        raise TypeError("Invalid evidence coverage.")

    if not isinstance(verification, dict):
        raise TypeError("Invalid verification result.")

    status = verification.get("status")

    allowed_statuses = {
        "verified",
        "insufficient_evidence",
        "conflicting_evidence",
    }

    if status not in allowed_statuses:
        raise ValueError("Unknown verification status.")

    claims = (
        generated_result.get("claims", [])
        if isinstance(generated_result, dict)
        else []
    )

    # Map verifier decisions by their explicit claim indices.
    claim_checks = verification.get("claim_checks") or []

    checks_by_index = {
        check["claim_index"]: check
        for check in claim_checks
        if isinstance(check, dict)
        and isinstance(check.get("claim_index"), int)
    }

    # Evidence metadata comes from the existing coverage
    # analyzer, which uses the same E1/E2 mapping as generation.
    evidence_by_id = {
        item["evidence_id"]: item
        for item in evidence_coverage.get("cited_evidence", [])
        if isinstance(item, dict)
        and isinstance(item.get("evidence_id"), str)
    }

    candidate_claims = []

    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            continue

        check = checks_by_index.get(index)

        verification_label = (
            check.get("status", "not_assessed")
            if check
            else "not_assessed"
        )

        verification_reason = (
            check.get("reason", "")
            if check
            else ""
        )

        evidence_ids = claim.get("evidence_ids", [])

        sources = [
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        ]

        candidate_claims.append({
            "claim_index": index,
            "text": claim.get("text", ""),
            "evidence_ids": evidence_ids,
            "sources": sources,
            "verification_label": verification_label,
            "verification_reason": verification_reason,
            "accepted_in_final_answer": (
                status == "verified"
                and verification_label == "supported"
            ),
        })

    if status == "verified":
        conclusion = verification.get("answer", "")
        report_type = "verified_answer"

    elif status == "conflicting_evidence":
        conclusion = (
            "The retrieved evidence contains an unresolved "
            "material conflict. No definitive legal "
            "conclusion is established."
        )
        report_type = "unresolved_conflict"

    else:
        conclusion = (
            "The available evidence is insufficient to "
            "establish a reliable answer."
        )
        report_type = "insufficient_evidence"

    return {
        "question_type": question_analysis.get("question_type"),
        "report_type": report_type,
        "final_status": status,
        "conclusion": conclusion,
        "candidate_claims": candidate_claims,
        "evidence_coverage": evidence_coverage,
        "verified_by_existing_pipeline": (
            status == "verified"
        ),
        "analysis_method": "deterministic",
    }
