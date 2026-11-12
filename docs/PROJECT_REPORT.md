# Legal Document Intelligence
## Technical Project Report

**Project category:** Applied Artificial Intelligence / Natural Language Processing

**Architecture:** Hybrid Retrieval-Augmented Generation (RAG)

**Implementation:** Python 3.11, Streamlit, Gemini, FAISS, BM25

**Project status:** Functional research prototype with documented deployment limitations

---

## Executive Summary

Legal Document Intelligence is an AI-assisted system for exploring and analyzing legal PDF documents through evidence-grounded question answering.

The application combines legal-aware document processing, dense semantic retrieval, BM25 keyword retrieval, Reciprocal Rank Fusion, language model generation, and citation verification.

Additional engineering features include multi-document case management, embedding checkpoint recovery, document identity tracking, and interactive evidence visualization.

A preliminary GDPR retrieval benchmark evaluated 16 answerable questions using 397 chunks extracted from an 88-page document.

The hybrid retrieval pipeline achieved Hit@5 of 1.0000 and MRR@5 of 0.7917, compared with BM25 scores of 0.7500 and 0.5073 respectively.

These results demonstrate improved retrieval performance on the selected benchmark. They do not establish general legal-answer accuracy.

---

## 1. Introduction

### 1.1 Background

Legal documents frequently contain extensive text, hierarchical sections, specialized terminology, and references distributed across multiple pages.

Traditional keyword search can retrieve exact terms but may fail when a question expresses the same legal concept using different wording.

Large language models can generate readable explanations, but responses may be unsupported or incorrectly attributed when source evidence is not explicitly supplied.

Retrieval-Augmented Generation addresses this challenge by retrieving relevant source material before generating an answer.

### 1.2 Problem Statement

The project investigates how to construct a legal document analysis workflow that:

- Preserves legal document structure
- Supports semantic and lexical evidence retrieval
- Generates answers using retrieved source material
- Maintains traceable document citations
- Supports multiple documents and case workspaces
- Handles interruptions during computationally expensive processing

### 1.3 Project Objectives

The primary objectives are:

1. Extract and structure content from legal PDF documents.
2. Preserve legal hierarchy and document identity.
3. Combine semantic and keyword retrieval.
4. Generate evidence-grounded answers.
5. Validate citation references and assess evidence support.
6. Provide an accessible interactive interface.
7. Implement reliability mechanisms for interrupted document processing.
8. Evaluate retrieval performance using a reproducible benchmark.

---

## 2. System Design

The application follows a modular hybrid RAG architecture.

The principal stages are:

1. PDF ingestion
2. Text and structure extraction
3. Legal-aware chunking
4. Embedding generation
5. FAISS and BM25 indexing
6. Hybrid retrieval
7. Evidence selection and optional reranking
8. Grounded answer generation
9. Citation and answer verification
10. Interactive result presentation

Detailed diagrams and module responsibilities are documented in [System Architecture](ARCHITECTURE.md).

---

## 3. Methodology

### 3.1 PDF Extraction

PyMuPDF is used to extract text and page-level information from PDF files.

The implementation includes support for identifying pages that may require OCR and utilities for processing extracted tables.

Extraction quality can vary depending on the PDF's structure and formatting.

### 3.2 Legal-Aware Chunking

The chunking pipeline identifies legal structural information, including articles, paragraphs, and other hierarchical markers where detectable.

Rather than treating the document as a single unstructured text sequence, the system creates chunks with associated metadata.

This metadata supports retrieval and citation attribution.

### 3.3 Document Identity

Document identifiers are propagated through the processing and retrieval pipeline.

This allows documents with identical filenames to remain distinguishable and supports multi-document evidence attribution.

### 3.4 Embedding Generation

Legal chunks are converted into numerical representations using the Gemini embedding model.

The configured embedding model is `gemini-embedding-001`, with output dimensionality of 768.

Embeddings are used to construct a FAISS vector index.

### 3.5 BM25 Retrieval

The BM25 implementation performs lexical retrieval based on query terms and indexed document content.

This complements semantic retrieval when exact legal terminology or distinctive references are important.

### 3.6 Hybrid Retrieval

Dense and sparse retrieval outputs are combined using Reciprocal Rank Fusion.

This approach integrates rankings from different retrieval methods without requiring their raw scores to share a common scale.

Optional reranking functionality is available for further evidence prioritization.

### 3.7 Grounded Answer Generation

The generation component organizes retrieved evidence into a context supplied to a Gemini language model.

The system is designed to produce responses tied to source evidence rather than relying solely on model knowledge.

### 3.8 Verification

The verification layer includes deterministic citation integrity checks and model-assisted answer verification.

The system also includes legal reasoning and evidence coverage analysis.

These components support traceability but do not guarantee legal correctness.

---

## 4. Implementation

### 4.1 Technology Stack

| Component | Technology |
|---|---|
| Application language | Python 3.11 |
| User interface | Streamlit |
| PDF extraction | PyMuPDF |
| OCR dependency | Tesseract OCR |
| Embeddings | Gemini Embedding |
| Dense retrieval | FAISS |
| Sparse retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Generation | Gemini via Google GenAI SDK |
| Reranking support | Sentence Transformers |
| Numerical operations | NumPy |
| Testing | pytest |

### 4.2 Multi-Document Management

The application maintains document and case registries.

Documents can be associated with case workspaces, allowing retrieval to operate across relevant document collections.

Document identity metadata helps preserve source attribution.

### 4.3 Reliability Features

The implementation includes:

- Batched embedding requests
- Incremental embedding checkpoints
- Atomic checkpoint writes
- Checkpoint fingerprint validation
- Embedding provenance validation
- Document processing locks
- Progress callbacks
- API retry safeguards

These mechanisms help manage interrupted processing and avoid unnecessary repeated embedding requests when valid checkpoints are available.

### 4.4 Deployment

The application is deployed through Streamlit Cloud.

Live model-dependent functionality requires Gemini API access and available quota.

The deployed application has encountered embedding API retry delays lasting several hours.

A subsequent safeguard was implemented to stop processing when the provider requests an excessive delay.

This prevents the application from waiting indefinitely but does not eliminate provider quota restrictions.

---

## 5. Experimental Evaluation

### 5.1 Evaluation Dataset

The initial benchmark uses the General Data Protection Regulation (GDPR).

| Dataset property | Value |
|---|---:|
| Extracted pages | 88 |
| Legal chunks | 397 |
| Benchmark questions | 20 |
| Answerable questions | 16 |
| Out-of-scope questions | 4 |

The questions and expected evidence labels were manually prepared and remain provisional.

### 5.2 Retrieval Methods

Two configurations were compared:

- BM25 lexical retrieval
- Hybrid FAISS + BM25 retrieval using Reciprocal Rank Fusion

The evaluation cutoff was K=5.

### 5.3 Evaluation Metrics

**Hit@5:** Proportion of evaluated questions for which an expected relevant chunk appears among the top five results.

**Recall@5:** Proportion of labeled relevant chunks retrieved within the top five results.

**MRR@5:** Mean Reciprocal Rank of the first expected relevant result, with ranks beyond five treated as misses.

Each answerable question has one expected relevant chunk, so Hit@5 and Recall@5 are numerically identical in this benchmark.

### 5.4 Results

| Metric | BM25 | Hybrid |
|---|---:|---:|
| Hit@5 | 0.7500 | **1.0000** |
| Recall@5 | 0.7500 | **1.0000** |
| MRR@5 | 0.5073 | **0.7917** |

### 5.5 Interpretation

BM25 retrieved the expected chunk within the top five for 12 of the 16 answerable questions.

Hybrid retrieval retrieved the expected chunk within the top five for all 16 answerable questions.

Four questions missed by BM25 were recovered by the hybrid pipeline.

The higher MRR@5 indicates that expected evidence generally appeared earlier in the hybrid ranking.

These findings support the usefulness of combining semantic and lexical retrieval on the selected GDPR benchmark.

### 5.6 Evaluation Limitations

The benchmark is not a comprehensive legal evaluation.

Important limitations include:

- Only 16 answerable questions contributed to retrieval metrics.
- Expected evidence labels are provisional.
- Each question has one labeled relevant chunk.
- Other legally relevant chunks may exist.
- Dense-backend success or failure was not recorded per question.
- Query embedding task type was not recorded in the saved report.
- Temporary retrieval fallback behavior may affect interpretation.
- Citation correctness was not measured.
- Generated-answer accuracy and factual grounding were not evaluated.
- Abstention behavior was not evaluated.
- Retrieval latency was not measured.

The four out-of-scope questions were excluded from the retrieval metrics.

Full benchmark details are available in [GDPR Evaluation Documentation](../data/evaluation/README.md).

---

## 6. Testing and Validation

The repository contains 14 automated test files.

At the latest validated checkpoint, 87 automated tests passed.

The tests cover selected behaviors involving:

- GDPR baseline consistency
- Document identity
- Ingestion reliability
- Embedding checkpoint recovery
- Embedding retry limits
- Retrieval behavior
- Generation and verification
- Streamlit progress integration

The original GDPR processing baseline preserved 397 chunks across 88 extracted pages during regression testing.

These results demonstrate selected functional and regression-test coverage.

They do not constitute comprehensive security testing, production load testing, or legal-answer validation.

---

## 7. Engineering Challenges

### 7.1 Preserving Legal Structure

Legal text cannot always be divided into arbitrary fixed-length segments without losing useful hierarchy.

The project therefore implements structure-aware chunking and metadata preservation.

### 7.2 Combining Retrieval Strategies

Lexical and semantic retrieval have different strengths.

Reciprocal Rank Fusion provides a practical method for combining their ranked outputs.

### 7.3 Multi-Document Source Attribution

Identical filenames can occur across document collections.

Document identity propagation reduces ambiguity when associating retrieved evidence with its source.

### 7.4 Interrupted Embedding Generation

Embedding generation requires external API calls and can fail because of rate limits or service interruptions.

Checkpoint-based processing supports recovery when valid checkpoint data remains available.

### 7.5 External API Quota Restrictions

The deployment encountered Gemini embedding retry delays of several hours.

A fail-fast safeguard prevents excessively long blocking waits.

This improves failure handling but does not guarantee that a document can be processed when API quota is exhausted.

---

## 8. Limitations

The application remains a research prototype.

Its principal limitations include:

1. Dependence on external Gemini API quota and availability.
2. Potential loss of locally stored state on ephemeral hosting.
3. Limited retrieval evaluation coverage.
4. No comprehensive legal-answer correctness benchmark.
5. No formal citation accuracy evaluation.
6. OCR and PDF parsing limitations for complex documents.
7. Demo-level authentication rather than production-grade access management.
8. Synchronous operations that can block application execution.
9. Large functions and modules that may require maintainability improvements.

Users must independently verify important legal conclusions.

---

## 9. Future Work

The following improvements are proposed for future development.

### 9.1 Persistent Storage

Introduce durable storage for uploaded documents, case metadata, embedding checkpoints, and retrieval indexes.

### 9.2 Background Processing

Move long-running ingestion and embedding tasks to asynchronous workers or a job queue.

### 9.3 Expanded Legal Evaluation

Develop a larger benchmark covering different legal domains, document types, and question categories.

### 9.4 Answer and Citation Evaluation

Measure citation correctness, factual grounding, abstention behavior, and unsupported-answer frequency.

### 9.5 Improved API Resilience

Introduce quota monitoring, structured failure classification, and operational alerts.

### 9.6 Production Security

Implement stronger authentication, authorization, and document access controls.

### 9.7 Code Quality

Refactor selected large functions, improve exception handling, and expand automated test coverage.

These are future plans and are not presented as completed features.

---

## 10. Conclusion

Legal Document Intelligence demonstrates a modular approach to AI-assisted legal document exploration using hybrid Retrieval-Augmented Generation.

The project integrates document extraction, legal-aware chunking, semantic and lexical retrieval, evidence-grounded generation, citation verification, multi-document management, and processing recovery mechanisms.

The initial GDPR benchmark shows improved retrieval performance for hybrid retrieval compared with BM25 alone.

However, retrieval performance does not automatically establish legal-answer correctness.

The project provides a functional foundation for further research into trustworthy, evidence-grounded legal document analysis.

---

## References and Project Resources

- [Project README](../README.md)
- [System Architecture](ARCHITECTURE.md)
- [GDPR Evaluation Documentation](../data/evaluation/README.md)
- [Source Code](../src/)
- [Automated Tests](../tests/)
- [Streamlit Application](https://legal-document-intelligence-demo.streamlit.app/)

**Disclaimer:** This application is an educational research prototype and does not provide professional legal advice.
