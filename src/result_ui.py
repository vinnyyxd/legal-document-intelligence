"""
Phase 17 — Legal research result presentation.

Displays existing RAG and verification metadata.
No retrieval, model calls, or verification changes.
"""

import streamlit as st


STATUS_LABELS = {
    "verified": "Verified Answer",
    "insufficient_evidence": "Insufficient Evidence",
    "conflicting_evidence": "Conflicting Evidence",
}

STATUS_EXPLANATIONS = {
    "verified": (
        "The automated verifier accepted the answer "
        "based on the retrieved evidence."
    ),
    "insufficient_evidence": (
        "The available evidence did not establish "
        "a fully supported answer."
    ),
    "conflicting_evidence": (
        "The verifier identified unresolved "
        "contradictions in the retrieved evidence."
    ),
}


def render_result_summary(result):
    """Display a concise verification overview."""

    if not isinstance(result, dict):
        return

    status = result.get("status", "unknown")
    confidence = result.get("confidence")

    verification = result.get("verification") or {}
    report = result.get("reasoning_report") or {}

    claim_checks = verification.get("claim_checks") or []

    supported_claims = sum(
        isinstance(check, dict)
        and check.get("status") == "supported"
        for check in claim_checks
    )

    total_claims = len(claim_checks)

    documents = result.get("documents") or []
    document_count = len(documents)

    st.markdown("### Research Verification Overview")

    if status == "verified":
        st.success(STATUS_EXPLANATIONS[status])
    elif status == "conflicting_evidence":
        st.warning(STATUS_EXPLANATIONS[status])
    elif status == "insufficient_evidence":
        st.info(STATUS_EXPLANATIONS[status])
    else:
        st.warning("The result has an unknown verification status.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Result",
        STATUS_LABELS.get(status, "Unknown"),
    )

    col2.metric(
        "Confidence",
        str(confidence) if confidence else "N/A",
    )

    col3.metric(
        "Case Documents",
        document_count,
    )

    col4.metric(
        "Supported Claims",
        f"{supported_claims}/{total_claims}",
    )

    if status != "verified":
        st.caption(
            "Individual supported claims do not mean "
            "the overall answer passed verification."
        )

    if report:
        conclusion = report.get("conclusion")

        if conclusion:
            with st.expander(
                "Reasoning conclusion",
                expanded=False,
            ):
                st.write(conclusion)

    st.caption(
        "Verification is automated. "
        "Review cited source passages and original "
        "PDF pages before relying on the result."
    )
