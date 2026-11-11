
import streamlit as st
from google import genai
from sentence_transformers import CrossEncoder

from src.multi_document_retrieval import answer_multiple_documents

from src.document_pipeline import ingest_document
from src.document_registry import list_documents
from src.document_structure import analyze_document_structure

from src.case_management import (
    create_case,
    get_case,
    list_cases,
    add_document_to_case,
    remove_document_from_case,
    get_case_documents,
    validate_case_for_query,
)



def get_gemini_client():
    """Initialize Gemini using Streamlit secrets."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "GEMINI_API_KEY is missing from Streamlit secrets."
        )

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is empty.")

    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner="Loading reranking model...")
def get_reranker():
    """Load the existing cross-encoder model once per app process."""
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def display_documents():
    """Display documents persisted by the Phase 10 registry."""

    records = list_documents()

    if isinstance(records, dict):
        records = list(records.values())

    if not records:
        st.info("No documents registered yet.")
        return

    for record in records:
        if not isinstance(record, dict):
            continue

        filename = record.get("filename", "Unknown document")
        document_id = record.get("document_id", "Unknown ID")
        status = record.get("status", "unknown")

        with st.container(border=True):
            st.markdown(f"**📄 {filename}**")
            st.caption(f"Document ID: {document_id}")

            if status == "ready":
                st.success("Ready for search")
            elif status == "failed":
                st.error("Processing failed")
                if record.get("error"):
                    st.caption(str(record["error"]))
            else:
                st.info(f"Processing status: {status}")


def display_rag_result(result):
    """Render the final verifier-approved RAG outcome."""

    status = result.get("status", "unknown")
    answer = result.get("answer", "")
    confidence = result.get("confidence")

    st.divider()
    st.subheader("Legal Research Result")

    if status == "verified":
        st.success("Answer verified against retrieved evidence")
    elif status == "conflicting_evidence":
        st.warning("Conflicting evidence detected")
    elif status == "insufficient_evidence":
        st.warning("Insufficient verified evidence")
    else:
        st.error("Unexpected verification status")
        return

    if confidence:
        st.caption(f"Confidence: {confidence}")

    st.markdown("#### Answer")
    st.write(answer)

    verification = result.get("verification") or {}
    integrity = verification.get("citation_integrity")

    if isinstance(integrity, dict):
        if integrity.get("valid") is True:
            st.caption("Citation integrity: Valid")
        elif integrity.get("valid") is False:
            st.error("Citation integrity check failed")

    # Only display retrieved passages as supporting evidence
    # when the final answer has actually been verified.
    if status == "verified":
        evidence = result.get("reranked_results") or []

        st.markdown("#### Retrieved Legal Evidence")

        if not evidence:
            st.info("No evidence passages available.")

        for index, item in enumerate(evidence, start=1):
            document = item.get("document", "Unknown document")
            document_id = item.get("document_id", "Legacy document")
            clause = item.get("clause_reference")

            if not clause:
                evidence_pages = item.get("page_numbers") or []

                if len(evidence_pages) == 1:
                    clause = f"PDF p. {evidence_pages[0]}"
                elif evidence_pages:
                    clause = (
                        "PDF pp. "
                        + ", ".join(map(str, evidence_pages))
                    )
                else:
                    clause = "Page unavailable"
            pages = item.get("page_numbers") or []
            chunk_id = item.get("chunk_id", "Unknown chunk")

            with st.expander(
                f"Evidence {index}: {document} — {clause}"
            ):
                st.caption(f"Document ID: {document_id}")
                st.caption(f"Chunk ID: {chunk_id}")
                st.caption(f"Pages: {pages}")
                st.write(item.get("text", ""))

    with st.expander("Verification Details"):
        if verification:
            st.json({
                "status": verification.get("status"),
                "confidence": verification.get("confidence"),
                "citation_integrity": integrity,
                "question_assessment": verification.get(
                    "question_assessment"
                ),
                "claim_checks": verification.get("claim_checks"),
            })
        else:
            st.info(
                "No independent verification was performed "
                "because the pipeline abstained early."
            )


def get_ready_documents():
    """Return ready documents indexed by unique document ID."""

    records = list_documents()

    if isinstance(records, dict):
        records = list(records.values())

    return {
        record["document_id"]: record
        for record in records
        if (
            isinstance(record, dict)
            and record.get("status") == "ready"
            and record.get("document_id")
        )
    }


st.set_page_config(
    page_title="Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.15);
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Legal Document Intelligence")

st.caption(
    "Multi-document legal research powered by "
    "hybrid retrieval, reranking and verified citations."
)

with st.sidebar:
    st.header("Document Workspace")

    uploaded_files = st.file_uploader(
        "Upload legal documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF documents.",
    )

    st.divider()
    st.subheader("Indexed Documents")

    display_documents()

    st.divider()
    st.caption("RAG Pipeline")
    st.markdown(
        "PDF → Chunks → Embeddings → "
        "FAISS + BM25 → Reranking → "
        "Generation → Verification"
    )


st.header("Legal Case Workspace")

with st.expander("Create a New Case", expanded=False):
    with st.form("create_case_form"):
        new_case_name = st.text_input(
            "Case name",
            placeholder="Example: Rental Agreement Review",
        )
        new_case_description = st.text_area(
            "Description (optional)",
            height=80,
        )
        create_clicked = st.form_submit_button(
            "Create Case",
            type="primary",
        )

    if create_clicked:
        try:
            created = create_case(
                name=new_case_name,
                description=new_case_description,
            )
            st.session_state["active_case_id"] = created["case_id"]
            st.success(f"Case created: {created['name']}")
            st.rerun()
        except Exception as exc:
            st.error(f"Case creation failed: {exc}")

available_cases = list_cases()

case_lookup = {
    case["case_id"]: case
    for case in available_cases
}

case_options = list(case_lookup)

previous_case_id = st.session_state.get("active_case_id")

if previous_case_id not in case_lookup:
    previous_case_id = case_options[0] if case_options else None

active_case_id = st.selectbox(
    "Active legal case",
    options=case_options,
    index=(
        case_options.index(previous_case_id)
        if previous_case_id in case_options
        else None
    ),
    format_func=lambda case_id: case_lookup[case_id]["name"],
    placeholder="Create a case to begin",
)

st.session_state["active_case_id"] = active_case_id

if active_case_id:
    active_case = get_case(active_case_id)
    assigned_ids = active_case["document_ids"]

    st.caption(
        f"Documents in case: {len(assigned_ids)}/5 "
        "(minimum 2 required for questions)"
    )

    all_documents = list_documents()

    if isinstance(all_documents, dict):
        all_documents = list(all_documents.values())

    available_document_lookup = {
        document["document_id"]: document
        for document in all_documents
        if isinstance(document, dict)
        and document.get("document_id")
    }

    attachable_ids = [
        document_id
        for document_id in available_document_lookup
        if document_id not in assigned_ids
    ]

    with st.expander("Assign Indexed Documents to Case"):
        selected_attachment_ids = st.multiselect(
            "Available documents",
            options=attachable_ids,
            format_func=lambda document_id: (
                available_document_lookup[document_id].get(
                    "filename", "Document"
                )
                + " — "
                + document_id[:8]
            ),
            key="case_document_attachment",
        )

        if st.button(
            "Add Documents to Case",
            disabled=not selected_attachment_ids,
        ):
            try:
                if len(assigned_ids) + len(selected_attachment_ids) > 5:
                    raise ValueError(
                        "A case cannot contain more than five PDFs."
                    )

                for document_id in selected_attachment_ids:
                    add_document_to_case(
                        active_case_id,
                        document_id,
                    )

                st.success("Documents assigned successfully.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if assigned_ids:
        st.markdown("**Documents assigned to this case:**")

        for document in get_case_documents(active_case_id):
            if not document:
                continue

            document_id = document["document_id"]

            name_column, remove_column = st.columns([5, 1])

            with name_column:
                st.write(
                    f"📄 {document.get('filename', 'Document')} "
                    f"— {document.get('status', 'unknown')}"
                )

            with remove_column:
                if st.button(
                    "Remove",
                    key=f"remove_{active_case_id}_{document_id}",
                ):
                    remove_document_from_case(
                        active_case_id,
                        document_id,
                    )
                    st.session_state.pop("last_rag_result", None)
                    st.session_state.pop("last_rag_query_key", None)
                    st.rerun()


tab_qa, tab_documents, tab_structure = st.tabs([
    "Ask Questions",
    "Document Management",
    "Document Structure",
])

with tab_qa:
    st.subheader("Ask your legal documents")

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What are the obligations "
            "for reporting a data breach?"
        ),
        height=110,
    )

    ready_documents = get_ready_documents()

    if active_case_id:
        active_case = get_case(active_case_id)
        selected_document_ids = [
            document_id
            for document_id in active_case["document_ids"]
            if document_id in ready_documents
        ]
    else:
        selected_document_ids = []

    st.caption(
        f"Searching {len(selected_document_ids)} ready documents "
        "from the active case."
    )

    current_query_key = (
        active_case_id,
        question.strip(),
        tuple(sorted(selected_document_ids)),
    )

    if (
        st.session_state.get("last_rag_query_key")
        != current_query_key
    ):
        st.session_state.pop("last_rag_result", None)

    search_enabled = (
        bool(question.strip())
        and bool(active_case_id)
        and 2 <= len(selected_document_ids) <= 5
    )

    search_clicked = st.button(
        "Search and Verify",
        type="primary",
        disabled=not search_enabled,
    )

    if search_clicked:
        try:
            validated_case = validate_case_for_query(
                active_case_id
            )
            selected_document_ids = validated_case["document_ids"]

            client = get_gemini_client()
            reranker_model = get_reranker()

            with st.spinner(
                "Retrieving, reranking and verifying legal evidence..."
            ):
                result = answer_multiple_documents(
                    client=client,
                    question=question.strip(),
                    document_ids=selected_document_ids,
                    reranker_model=reranker_model,
                )

            st.session_state["last_rag_result"] = result
            st.session_state["last_rag_query_key"] = current_query_key

            if result.get("status") == "verified":
                st.success("Answer verified successfully.")
            elif result.get("status") in {
                "insufficient_evidence",
                "conflicting_evidence",
            }:
                st.info(
                    "Analysis completed. Review the verification result."
                )
            else:
                st.warning("Unexpected RAG result status.")

        except Exception as exc:
            st.session_state.pop("last_rag_result", None)
            st.session_state.pop("last_rag_query_key", None)
            st.error(f"RAG pipeline failed: {exc}")

    if "last_rag_result" in st.session_state:
        display_rag_result(
            st.session_state["last_rag_result"]
        )

    if not ready_documents:
        st.warning(
            "No indexed documents are ready. "
            "Upload and process a PDF first."
        )

with tab_documents:
    st.subheader("Document Management")

    display_documents()

    current_case_count = (
        len(get_case(active_case_id)["document_ids"])
        if active_case_id
        else 0
    )

    remaining_slots = 5 - current_case_count

    st.caption(
        f"Remaining PDF slots in active case: {remaining_slots}"
    )

    upload_count = len(uploaded_files) if uploaded_files else 0

    if upload_count > remaining_slots:
        st.warning(
            f"Only {remaining_slots} additional PDFs can be "
            "added to this case."
        )

    process_clicked = st.button(
        "Process Uploaded Documents",
        type="primary",
        disabled=(
            not active_case_id
            or upload_count == 0
            or upload_count > remaining_slots
        ),
    )

    if process_clicked:
        st.session_state.pop("last_rag_result", None)
        st.session_state.pop("last_rag_query_key", None)

        try:
            if not active_case_id:
                raise ValueError("Create or select a case first.")

            remaining_slots = (
                5 - len(get_case(active_case_id)["document_ids"])
            )

            if not 1 <= upload_count <= remaining_slots:
                raise ValueError(
                    "Upload exceeds the remaining case capacity."
                )

            client = get_gemini_client()

            for uploaded in uploaded_files:
                with st.status(
                    f"Processing {uploaded.name}",
                    expanded=True,
                ) as status:

                    result = ingest_document(
                        client=client,
                        filename=uploaded.name,
                        pdf_bytes=uploaded.getvalue(),
                    )

                    if active_case_id:
                        document_id = result.get("document_id")

                        if not document_id:
                            raise RuntimeError(
                                "Ingestion result missing document_id."
                            )

                        add_document_to_case(
                            active_case_id,
                            document_id,
                        )

                    status.update(
                        label=f"Processed {uploaded.name}",
                        state="complete",
                    )

                    st.success(
                        f"Document processed: {uploaded.name}"
                    )

                    st.json(result)

        except Exception as exc:
            st.error(
                f"Document processing failed: {exc}"
            )

    if uploaded_files:
        st.write("Selected files:")
        for uploaded in uploaded_files:
            st.write(f"📄 {uploaded.name}")


# ==================================================
# PHASE 14 — DOCUMENT STRUCTURE VIEWER
# ==================================================

with tab_structure:

    st.subheader("Document Structure Explorer")

    if not active_case_id:
        st.info("Create or select a case first.")

    else:
        case_documents = get_case_documents(active_case_id)

        document_lookup = {
            document["document_id"]: document
            for document in case_documents
            if document
            and document.get("status") == "ready"
        }

        if not document_lookup:
            st.info(
                "No processed documents are available "
                "in this case."
            )

        else:
            selected_structure_document = st.selectbox(
                "Select a document",
                options=list(document_lookup),
                format_func=lambda document_id: (
                    document_lookup[document_id].get(
                        "filename", "Document"
                    )
                ),
                key="structure_document_selector",
            )

            try:
                with st.spinner(
                    "Analyzing document structure..."
                ):
                    structure = analyze_document_structure(
                        selected_structure_document
                    )

                # ----------------------------------
                # Page statistics
                # ----------------------------------

                st.markdown("### Document Overview")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Pages",
                    structure["page_count"],
                )

                col2.metric(
                    "Articles",
                    structure["article_count"],
                )

                col3.metric(
                    "Tables",
                    structure["table_count"],
                )

                col4.metric(
                    "Potentially Scanned",
                    structure["scanned_page_count"],
                )

                st.caption(
                    f"Readable pages: "
                    f"{structure['readable_page_count']} | "
                    f"Total words: {structure['total_words']:,}"
                )

                # ----------------------------------
                # Legal hierarchy
                # ----------------------------------

                st.markdown("### Legal Structure")

                hierarchy_col1, hierarchy_col2 = st.columns(2)

                with hierarchy_col1:
                    st.metric(
                        "Chapters",
                        structure["chapter_count"],
                    )

                with hierarchy_col2:
                    st.metric(
                        "Sections",
                        structure["section_count"],
                    )

                for heading, key in [
                    ("Chapters", "chapters"),
                    ("Sections", "sections"),
                    ("Articles", "articles"),
                ]:
                    items = structure[key]

                    with st.expander(
                        f"{heading} ({len(items)})"
                    ):
                        if items:
                            st.dataframe(
                                items,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info(
                                f"No {heading.lower()} detected."
                            )

                # ----------------------------------
                # Tables
                # ----------------------------------

                st.markdown("### Detected Tables")

                if structure["tables"]:
                    st.dataframe(
                        structure["tables"],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        "No tables detected using "
                        "PyMuPDF table detection."
                    )

                if structure["table_detection_warnings"]:
                    with st.expander(
                        "Table Detection Warnings"
                    ):
                        st.json(
                            structure["table_detection_warnings"]
                        )

                # ----------------------------------
                # Scanned pages
                # ----------------------------------

                st.markdown("### Scanned Page Detection")

                if structure["scanned_pages"]:
                    st.warning(
                        "Potentially scanned or text-light "
                        "pages detected: "
                        + ", ".join(
                            map(
                                str,
                                structure["scanned_pages"],
                            )
                        )
                    )

                    st.caption(
                        "These pages may require OCR. "
                        "The current pipeline does not "
                        "automatically extract image text."
                    )

                else:
                    st.success(
                        "No potentially scanned pages detected."
                    )

            except Exception as exc:
                st.error(
                    f"Document structure analysis failed: {exc}"
                )
