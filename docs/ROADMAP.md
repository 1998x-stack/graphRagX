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

## P1C — graph and retrieval quality

Implemented in V2.3:

- normalized-exact entity alias resolution
- entity/relationship mention preservation
- entity/relationship description summarization
- auditable graph pruning
- optional claim/covariate extraction
- claim-aware Local/DRIFT evidence
- index quality reports
- Basic/Local/Global/DRIFT evaluation harness
- deterministic source-recall and answer-term metrics

## P1D — original paper fidelity improvements (in progress)

Implemented on the paper-alignment branch:
- projected community cuts keep unsplit ancestors and reject invalid partitions
- default complete-report Global coverage with deterministic shuffled map batches
- tokenizer-aware fair context allocation and explicit resource-limit failures
- iterative fail-closed extraction, preventing false-success empty knowledge graphs
- text-unit Map/Reduce TS control arm
- six-condition C0/C1/C2/C3/TS/SS experiment harness
- optional token-based 600/100 chunking with source character/token offsets
- optional blind repeated pairwise judging over four paper criteria
- regression tests for coverage, extraction validity, TS, and pairwise evaluation
- detailed Section 3/4 paper-to-code parity matrix

Still outstanding (see `docs/PAPER_2404_16130_ALIGNMENT.md`):
- tokenizer/model/input-hash manifests, and evidence-grounded citations
- claim-inclusive and importance-ranked community report context
- persona-based benchmark question generation
- validated six-condition study with strict source, model and prompt-budget parity
- statistical confidence intervals, answer-level claim evaluation, source faithfulness

## P1E — next fidelity layer

- learned/fuzzy entity resolution with merge confidence
- dynamic community-level selection
- graph-aware reranking
- richer citation rendering
- conversation-history-aware query planning
- optional external judge metrics with calibration

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
