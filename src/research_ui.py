"""
Phase 17 — Legal research interface components.

Presentation-only helpers.
No retrieval, verification, or Gemini calls.
"""

import streamlit as st


EXAMPLE_QUESTIONS = {
    "Direct lookup": (
        "What obligations does the document impose "
        "on the responsible party?"
    ),
    "Comparison": (
        "How do the selected documents differ "
        "in their reporting requirements?"
    ),
    "Conflict analysis": (
        "Do the selected documents contain "
        "conflicting requirements or obligations?"
    ),
}


def render_research_intro(
    active_case_id,
    case_documents,
):
    """Display research guidance and case context."""

    st.markdown(
        "Ask questions across your selected legal documents. "
        "Answers are generated from retrieved passages "
        "and checked by a separate verification step."
    )

    if not active_case_id:
        st.info(
            "Select or create a legal case before "
            "asking questions."
        )
        return

    documents = [
        document
        for document in (case_documents or [])
        if isinstance(document, dict)
    ]

    ready_documents = [
        document
        for document in documents
        if document.get("status") == "ready"
    ]

    st.caption(
        f"Research scope: {len(ready_documents)} "
        "ready document(s) in the active case."
    )

    if ready_documents:
        with st.expander(
            "Documents included in research",
            expanded=False,
        ):
            for document in ready_documents:
                filename = document.get(
                    "filename",
                    "Unknown document",
                )

                st.write(f"📄 {filename}")

    with st.expander(
        "Example legal questions",
        expanded=False,
    ):
        for question_type, example in (
            EXAMPLE_QUESTIONS.items()
        ):
            st.markdown(
                f"**{question_type}**"
            )
            st.write(example)

    st.caption(
        "Research assistance only. "
        "Generated answers are not a substitute "
        "for professional legal advice."
    )


def render_verification_guide():
    """Explain the possible verification outcomes."""

    with st.expander(
        "How answer verification works",
        expanded=False,
    ):
        st.markdown(
            "**Verified:** Retrieved evidence supports "
            "the generated claims and the verifier "
            "finds the answer sufficient."
        )

        st.markdown(
            "**Insufficient evidence:** The documents "
            "do not establish a fully supported answer."
        )

        st.markdown(
            "**Conflicting evidence:** The verifier "
            "identifies unresolved contradictory evidence."
        )

        st.caption(
            "Verification is automated and may be "
            "incorrect. Always inspect original sources."
        )
