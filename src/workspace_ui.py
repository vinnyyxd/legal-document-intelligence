"""
Phase 17 — Streamlit workspace dashboard.

Presentation-only module.
No Gemini calls, retrieval, or document mutation.
"""

import streamlit as st


def render_workspace_dashboard(
    active_case_id,
    case_record,
    case_documents,
):
    """Render an overview of the selected legal case."""

    st.markdown("### Workspace Overview")

    if not active_case_id or not case_record:
        st.info(
            "Create or select a legal case to begin."
        )
        return

    documents = [
        document
        for document in (case_documents or [])
        if isinstance(document, dict)
    ]

    attached_count = len(
        case_record.get("document_ids") or []
    )

    ready_count = sum(
        document.get("status") == "ready"
        for document in documents
    )

    failed_count = sum(
        document.get("status") == "failed"
        for document in documents
    )

    maximum_documents = 5

    if attached_count == 0:
        research_status = "No documents"
    elif failed_count:
        research_status = "Needs attention"
    elif ready_count == attached_count:
        research_status = "Ready"
    else:
        research_status = "Processing"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Active Case",
        "Selected",
    )

    col2.metric(
        "Case Documents",
        f"{attached_count}/{maximum_documents}",
    )

    col3.metric(
        "Research Status",
        research_status,
    )

    st.progress(
        min(attached_count / maximum_documents, 1.0)
    )

    case_name = case_record.get(
        "name",
        "Unnamed Case",
    )

    st.caption(f"Current case: {case_name}")

    if research_status == "Ready":
        st.success(
            "Your case documents are ready "
            "for legal research."
        )

    elif research_status == "Needs attention":
        st.warning(
            "One or more case documents failed processing."
        )

    elif research_status == "Processing":
        st.info(
            "Some case documents are not yet ready."
        )

    else:
        st.info(
            "Attach processed documents to start "
            "researching this case."
        )
