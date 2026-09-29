# graphRagX Roadmap

## P0 — correctness and reproducibility (implemented in V2 foundation)

- deterministic offline embedding provider
- provider selection from configuration
- persisted retrieval embeddings
- source text units in Local Search
- directed typed relation graph
- relation deduplication
- workflow error short-circuiting
- safe index IDs and atomic JSON writes
- input/resource limits
- Basic vector-RAG comparison mode
- tests and CI

## P1 — GraphRAG fidelity

- hierarchical Leiden communities and bottom-up reports
- entity/relationship description summarization across mentions
- map/reduce Global Search with token budgets
- DRIFT query mode
- claim/covariate extraction
- tokenizer-aware context packing and provenance ranking

## P2 — indexing platform

- content-addressed documents and incremental updates
- index manifests and migrations
- Parquet/DuckDB storage adapter
- vector database adapter
- resumable jobs and checkpoints
- document parsers and metadata-preserving ingestion

## P3 — evaluation and operations

- retrieval evaluation set and answer-level regression tests
- cost, latency, token, and cache metrics
- OpenTelemetry/LangSmith-compatible tracing hooks
- API auth, tenant isolation, quotas, and rate limits
- Docker image and deployment templates
- benchmark report comparing Basic / Local / Global / DRIFT
