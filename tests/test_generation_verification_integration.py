
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.generation import generate_grounded_answer
from src.verification import verify_generated_answer


QUESTION = "What records must the controller maintain?"

EVIDENCE = [{
    "chunk_id": "doc_a_chunk_1",
    "document_id": "doc_a",
    "document": "Contract.pdf",
    "clause_reference": "Article 4(1)",
    "page_numbers": [2],
    "text": "The controller must maintain records.",
}]


def generator_payload(
    status="answered",
    claim_text="The controller must maintain records.",
    evidence_ids=None,
):
    if evidence_ids is None:
        evidence_ids = ["E1"]

    return {
        "status": status,
        "claims": (
            [{
                "text": claim_text,
                "evidence_ids": evidence_ids,
            }]
            if status == "answered"
            else []
        ),
        "insufficient_evidence_reason": (
            "The supplied passages do not answer the question."
            if status != "answered"
            else ""
        ),
    }


def verifier_payload(
    question_status="sufficient",
    claim_status="supported",
    claim_count=1,
):
    return {
        "question_status": question_status,
        "question_reason": "Mock verification assessment.",
        "missing_aspects": [],
        "claim_checks": [
            {
                "claim_index": i,
                "status": claim_status,
                "reason": "Mock claim assessment.",
            }
            for i in range(1, claim_count + 1)
        ],
    }


def mock_response(payload):
    return SimpleNamespace(text=json.dumps(payload))


def run_pipeline(
    generated_payload,
    verified_payload,
):
    responses = [
        mock_response(generated_payload),
        mock_response(verified_payload),
    ]

    with (
        patch(
            "src.generation.generate_with_retry",
            return_value=responses[0],
        ) as generator_mock,
        patch(
            "src.verification.generate_with_retry",
            return_value=responses[1],
        ) as verifier_mock,
    ):

        generated = generate_grounded_answer(
            client=None,
            question=QUESTION,
            retrieved_results=EVIDENCE,
        )

        verified = verify_generated_answer(
            client=None,
            question=QUESTION,
            generated_result=generated,
            retrieved_results=EVIDENCE,
        )

    generator_mock.assert_called_once()
    verifier_mock.assert_called_once()

    return generated, verified


def test_supported_answer_verified():

    generated, verified = run_pipeline(
        generator_payload(),
        verifier_payload(),
    )

    assert generated["status"] == "answered"
    assert verified["status"] == "verified"
    assert verified["confidence"] == "HIGH"
    assert verified["citation_integrity"]["valid"]
    assert "[Contract.pdf, Article 4(1), PDF p. 2]" in verified["answer"]


def test_unsupported_claim_abstains():

    _, verified = run_pipeline(
        generator_payload(
            claim_text="The controller must delete all records."
        ),
        verifier_payload(
            claim_status="unsupported"
        ),
    )

    assert verified["status"] == "insufficient_evidence"
    assert verified["confidence"] == "LOW"
    assert "not contain enough" in verified["answer"]


def test_conflicting_evidence_rejected():

    _, verified = run_pipeline(
        generator_payload(),
        verifier_payload(
            question_status="conflicting",
            claim_status="conflicting",
        ),
    )

    assert verified["status"] == "conflicting_evidence"
    assert verified["confidence"] == "LOW"
    assert "conflicting" in verified["answer"].lower()


def test_insufficient_evidence_abstains():

    _, verified = run_pipeline(
        generator_payload(
            status="insufficient_evidence"
        ),
        verifier_payload(
            question_status="insufficient",
            claim_count=0,
        ),
    )

    assert verified["status"] == "insufficient_evidence"
    assert verified["confidence"] == "INSUFFICIENT EVIDENCE"


def test_hallucinated_evidence_id_rejected():

    with patch(
        "src.generation.generate_with_retry",
        return_value=mock_response(
            generator_payload(
                evidence_ids=["E999"]
            )
        ),
    ):

        with pytest.raises(
            ValueError,
            match="Unknown evidence ID",
        ):
            generate_grounded_answer(
                client=None,
                question=QUESTION,
                retrieved_results=EVIDENCE,
            )


def test_incomplete_verifier_claim_checks_rejected():

    generated_payload = generator_payload()

    with (
        patch(
            "src.generation.generate_with_retry",
            return_value=mock_response(generated_payload),
        ),
        patch(
            "src.verification.generate_with_retry",
            return_value=mock_response(
                verifier_payload(claim_count=0)
            ),
        ),
    ):

        generated = generate_grounded_answer(
            client=None,
            question=QUESTION,
            retrieved_results=EVIDENCE,
        )

        with pytest.raises(
            ValueError,
            match="did not assess every generated claim",
        ):
            verify_generated_answer(
                client=None,
                question=QUESTION,
                generated_result=generated,
                retrieved_results=EVIDENCE,
            )
