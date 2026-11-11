# GDPR Retrieval Evaluation

## Dataset

- Source: Original GDPR PDF
- Baseline: 397 legal chunks
- Benchmark: 20 provisional questions
- Answerable retrieval questions: 16
- Out-of-scope questions: 4

Expected evidence IDs are linked to existing GDPR chunks.

The benchmark is a small manually designed starter set,
not a comprehensive legal evaluation dataset.

## Retrieval Methods

- BM25 keyword retrieval
- Hybrid FAISS + BM25 retrieval
- Reciprocal Rank Fusion (RRF)

The hybrid evaluation uses the existing production
retrieval implementation.

## Metrics

Metrics are calculated at K=5.

| Metric | BM25 | Hybrid |
|---|---:|---:|
| Hit@5 | 0.7500 | 1.0000 |
| Recall@5 | 0.7500 | 1.0000 |
| MRR@5 | 0.5073 | 0.7917 |

Each answerable benchmark question currently has one
expected relevant chunk.

Consequently, Hit@5 and Recall@5 are numerically
identical for this benchmark.

## Interpretation

Hybrid retrieval returned the expected chunk within
the top five for all 16 answerable questions.

Four questions missed by BM25 at K=5 were recovered
by the hybrid pipeline.

These scores are specific to this small benchmark.
They do not establish general legal-answer accuracy.

## Limitations

- Benchmark labels are provisional.
- Only 16 answerable questions were evaluated.
- Each answerable question has one expected chunk.
- Other legally relevant chunks may exist.
- The report does not record dense-backend success
  or failure for each question.
- The report does not record the query embedding
  task type.
- The hybrid pipeline includes temporary backend
  fallback behavior.
- Citation correctness was not evaluated.
- Answer generation and factual grounding were
  not evaluated.
- The four out-of-scope questions were not included
  in retrieval metrics.
- Abstention behavior was not evaluated.
- Retrieval latency was not measured.

## Reproducibility

Benchmark:
`gdpr_retrieval_benchmark.json`

Measured report:
`gdpr_retrieval_report.json`

Metric implementation:
`src/evaluation.py`

The saved report supports deterministic metric
recalculation without additional Gemini API calls.

Reproducing live retrieval requires the original
GDPR chunks, FAISS index, embedding model,
BM25 implementation, and Gemini API access.
