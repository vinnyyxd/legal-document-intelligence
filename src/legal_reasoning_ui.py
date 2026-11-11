"""
Streamlit presentation for Phase 16 legal reasoning.

This module displays existing pipeline results.
It does not perform retrieval or call Gemini.
"""

import streamlit as st


def render_legal_reasoning_explorer(result):
    """Display question analysis and verified claim assessments."""

    if not isinstance(result, dict):
        return

    analysis = result.get("question_analysis") or {}
    coverage = result.get("evidence_coverage") or {}
    report = result.get("reasoning_report") or {}
    verification = result.get("verification") or {}

    if not analysis and not report:
        return

    with st.expander(
        "Legal Reasoning Explorer",
        expanded=False,
    ):
        st.markdown("#### Question Analysis")

        question_type = analysis.get(
            "question_type",
            "unknown",
        )

        st.write(
            "Question type:",
            question_type.replace("_", " ").title(),
        )

        st.caption(
            "Question classification is heuristic. "
            "It does not determine legal correctness."
        )

        st.markdown("#### Evidence Coverage")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Retrieved Documents",
            coverage.get("retrieved_document_count", 0),
        )

        col2.metric(
            "Cited Documents",
            coverage.get("cited_document_count", 0),
        )

        col3.metric(
            "Cited Passages",
            coverage.get("cited_evidence_count", 0),
        )

        warning = coverage.get("coverage_warning")

        if warning:
            st.warning(warning)

        st.markdown("#### Verification Summary")

        final_status = report.get(
            "final_status",
            result.get("status", "unknown"),
        )

        st.write(
            "Final status:",
            final_status.replace("_", " ").title(),
        )

        question_assessment = (
            verification.get("question_assessment") or {}
        )

        reason = question_assessment.get("reason")

        if reason:
            st.write("Verifier explanation:", reason)

        missing_aspects = question_assessment.get(
            "missing_aspects",
            [],
        )

        if missing_aspects:
            st.markdown("**Missing aspects**")

            for aspect in missing_aspects:
                st.write("-", aspect)

        st.markdown("#### Claim-Level Verification")

        claims = report.get("candidate_claims") or []

        if not claims:
            st.info(
                "No claim-level verification details "
                "are available for this result."
            )

        for claim in claims:
            index = claim.get("claim_index", "?")
            label = claim.get(
                "verification_label",
                "not_assessed",
            )

            accepted = claim.get(
                "accepted_in_final_answer",
                False,
            )

            with st.container(border=True):
                st.markdown(
                    f"**Claim {index} — "
                    f"{label.replace('_', ' ').title()}**"
                )

                st.write(claim.get("text", ""))

                if accepted:
                    st.success(
                        "Accepted in the verified answer."
                    )
                else:
                    st.caption(
                        "Not accepted as a verified "
                        "final-answer claim."
                    )

                explanation = claim.get(
                    "verification_reason",
                )

                if explanation:
                    st.write(
                        "Verification explanation:",
                        explanation,
                    )

                sources = claim.get("sources") or []

                if sources:
                    st.markdown("**Cited sources**")

                    for source in sources:
                        document = source.get(
                            "document",
                            "Unknown document",
                        )

                        pages = source.get(
                            "page_numbers",
                            [],
                        )

                        evidence_id = source.get(
                            "evidence_id",
                            "?",
                        )

                        st.caption(
                            f"{evidence_id} | "
                            f"{document} | "
                            f"Pages: {pages}"
                        )

        if final_status == "conflicting_evidence":
            st.warning(
                "The verifier identified unresolved "
                "conflicting evidence. No definitive "
                "conclusion should be drawn."
            )

        elif final_status == "insufficient_evidence":
            st.info(
                "The evidence did not establish a "
                "fully verified answer."
            )
