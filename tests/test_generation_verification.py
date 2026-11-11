
from copy import deepcopy
from unittest.mock import patch

import pytest

from src.generation import (
    build_evidence_context,
    validate_generated_output,
    format_cited_answer,
)
from src.verification import (
    validate_citation_integrity,
    classify_confidence,
    verify_generated_answer,
)


def evidence():
    return [{
        "chunk_id": "doc_a_chunk_1",
        "document_id": "doc_a",
        "document": "Contract.pdf",
        "clause_reference": "Article 4(1)",
        "page_numbers": [2],
        "text": "The controller must maintain records.",
    }]


def generated():
    context, evidence_map = build_evidence_context(
        evidence()
    )

    return {
        "status": "answered",
        "claims": [{
            "text": "The controller must maintain records.",
            "evidence_ids": ["E1"],
        }],
        "answer": "The controller must maintain records.",
        "evidence_map": evidence_map,
    }


# --------------------------------------------------
# Evidence construction
# --------------------------------------------------

def test_evidence_preserves_document_identity():

    context, evidence_map = build_evidence_context(
        evidence()
    )

    assert "E1" in context
    assert evidence_map["E1"]["document_id"] == "doc_a"
    assert evidence_map["E1"]["chunk_id"] == "doc_a_chunk_1"


def test_evidence_rejects_missing_pages():

    items = evidence()
    items[0]["page_numbers"] = []

    with pytest.raises(ValueError, match="Missing page"):
        build_evidence_context(items)


def test_unnumbered_evidence_uses_pdf_page():

    items = evidence()
    items[0]["clause_reference"] = None

    _, evidence_map = build_evidence_context(items)

    assert evidence_map["E1"]["clause_reference"] == "PDF p. 2"


# --------------------------------------------------
# Generation validation
# --------------------------------------------------

def test_valid_generated_claim():

    result = generated()

    assert validate_generated_output(
        {
            "status": result["status"],
            "claims": result["claims"],
        },
        result["evidence_map"],
    )


def test_unknown_evidence_rejected():

    result = generated()

    payload = {
        "status": "answered",
        "claims": [{
            "text": "Unsupported statement.",
            "evidence_ids": ["E999"],
        }],
    }

    with pytest.raises(ValueError, match="Unknown evidence"):
        validate_generated_output(
            payload,
            result["evidence_map"],
        )


def test_answer_without_claims_rejected():

    result = generated()

    with pytest.raises(ValueError, match="at least one claim"):
        validate_generated_output(
            {
                "status": "answered",
                "claims": [],
            },
            result["evidence_map"],
        )


# --------------------------------------------------
# Citation integrity
# --------------------------------------------------

def test_valid_citation_integrity():

    result = validate_citation_integrity(
        generated(),
        evidence(),
    )

    assert result["valid"]
    assert result["claims_checked"] == 1


def test_cross_document_citation_rejected():

    result = generated()
    result["evidence_map"]["E1"]["document_id"] = "doc_b"

    integrity = validate_citation_integrity(
        result,
        evidence(),
    )

    assert not integrity["valid"]
    assert any(
        "identity mismatch" in error.lower()
        for error in integrity["errors"]
    )


def test_incomplete_document_identity_rejected():

    result = generated()
    del result["evidence_map"]["E1"]["document_id"]

    integrity = validate_citation_integrity(
        result,
        evidence(),
    )

    assert not integrity["valid"]
    assert any(
        "incomplete document identity" in error.lower()
        for error in integrity["errors"]
    )


def test_wrong_page_rejected():

    result = generated()
    result["evidence_map"]["E1"]["page_numbers"] = [99]

    integrity = validate_citation_integrity(
        result,
        evidence(),
    )

    assert not integrity["valid"]
    assert any(
        "page mismatch" in error.lower()
        for error in integrity["errors"]
    )


# --------------------------------------------------
# Verification and confidence
# --------------------------------------------------

def test_invalid_citation_abstains_without_llm():

    result = generated()
    result["evidence_map"]["E1"]["document_id"] = "doc_b"

    with patch(
        "src.verification.generate_with_retry"
    ) as mocked_llm:

        verified = verify_generated_answer(
            client=None,
            question="What must the controller maintain?",
            generated_result=result,
            retrieved_results=evidence(),
        )

        mocked_llm.assert_not_called()

    assert verified["status"] == "insufficient_evidence"
    assert not verified["citation_integrity"]["valid"]


def test_confidence_classification():

    assert classify_confidence(
        "sufficient",
        [{"status": "supported"}],
        True,
    ) == "HIGH"

    assert classify_confidence(
        "sufficient",
        [{"status": "partially_supported"}],
        True,
    ) == "MEDIUM"

    assert classify_confidence(
        "conflicting",
        [{"status": "supported"}],
        True,
    ) == "LOW"

    assert classify_confidence(
        "insufficient",
        [],
        True,
    ) == "INSUFFICIENT EVIDENCE"
