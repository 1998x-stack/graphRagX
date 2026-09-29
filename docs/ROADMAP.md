# graphRagX Roadmap

## P0 — correctness and reproducibility

Implemented in V2:

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

## P1A — GraphRAG fidelity

Implemented in V2.1:

- hierarchical broad-to-fine communities
- optional Leiden backend
- bottom-up community reports
- community-level-aware Global Search
- map/reduce Global Search
- rated intermediate evidence points
- tokenizer-aware context budgets
- Local source provenance reranking

## P1B — iterative DRIFT

Implemented in V2.2:

- DRIFT query mode
- multi-fold community primer
- ranked follow-up generation
- iterative local graph exploration
- confidence-gated branch expansion
- query/follow-up deduplication
- explicit evidence ledger
- parent/depth trace hierarchy
- max depth and max action termination
- token-budgeted final reduce
- API provenance for DRIFT evidence

## P1C — next fidelity layer

- entity/relationship description summarization across mentions
- claim/covariate extraction
- graph pruning configuration
- dynamic community selection
- richer citation rendering
- conversation-history-aware query planning

## P2 — indexing platform

- content-addressed documents and incremental updates
- index manifests and migrations
- Parquet/DuckDB storage adapter
- vector database adapter
- resumable jobs and checkpoints
- document parsers and metadata-preserving ingestion

## P3 — evaluation and operations

- retrieval evaluation set and answer-level regression tests
- DRIFT traversal quality metrics
- cost, latency, token, and cache metrics
- OpenTelemetry/LangSmith-compatible tracing hooks
- API auth, tenant isolation, quotas, and rate limits
- Docker image and deployment templates
- benchmark report comparing Basic / Local / Global / DRIFT
