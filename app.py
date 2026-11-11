
import streamlit as st
from google import genai
from sentence_transformers import CrossEncoder

from src.multi_document_retrieval import answer_multiple_documents

from src.document_pipeline import ingest_document
from src.document_registry import list_documents


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

tab_qa, tab_documents = st.tabs([
    "Ask Questions",
    "Document Management",
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

    selected_document_ids = st.multiselect(
        "Select documents to search",
        options=list(ready_documents),
        format_func=lambda document_id: (
            f"{ready_documents[document_id].get('filename', 'Document')}"
            f" — {document_id[:8]}"
        ),
        help="Only successfully indexed documents are available.",
    )

    current_query_key = (
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
        and bool(selected_document_ids)
    )

    search_clicked = st.button(
        "Search and Verify",
        type="primary",
        disabled=not search_enabled,
    )

    if search_clicked:
        try:
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

    process_clicked = st.button(
        "Process Uploaded Documents",
        type="primary",
        disabled=not uploaded_files,
    )

    if process_clicked:
        st.session_state.pop("last_rag_result", None)
        st.session_state.pop("last_rag_query_key", None)

        try:
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
