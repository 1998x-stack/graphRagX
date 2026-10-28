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
- citation-grounded report schema, prompt budgets and full claim verification (basic claim inclusion and graph-degree importance sorting are implemented)
- persona-based benchmark question generation
- validated six-condition study with strict source, model and prompt-budget parity
- manifest completeness for tokenizer, prompt templates, provider versions and per-stage costs (source SHA-256/model/chunking information implemented)
- statistical confidence intervals, answer-level claim evaluation, source faithfulness

## P1D2 — Hugging Face multidimensional evaluation (implemented on PR #5)

- Version-pinned streaming HF adapters: HotpotQA, MuSiQue, Multi-News, QASPER
- Stable document and gold evidence IDs, fixture SHA-256 manifests
- Per-question frozen-index Basic/Local/Global/DRIFT/TS and C0-C3 runs
- QA EM/token F1, summary ROUGE-L, document evidence P/R/MRR/nDCG
- Graph density/connectivity/community coverage and source link diagnostics
- Query latency average/P95, index time, failure rate, per-mode slices
- Document reorder, synthetic distractors and source-removal stress variants
- Paired question-level bootstrap effects/95% confidence intervals
- Multi-News agreement guard and safe static-file ingestion
- Tests + detailed `docs/HF_EVALUATION_PROTOCOL.md`

Remaining before a scientific performance report:
- Actual HF dataset downloads and real model runs; CI is offline only
- Measured cost/tokens/throughput, factuality/hallucination verification
- Annotated graph entity/relation gold data and citation faithfulness checks
- End-to-end open-corpus retrieval with shared multi-question index
- Human-reviewed global question set, multiple seeds and prompt/model parity
- Paper-cluster bootstrap, Holm correction, judge calibration and plots

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
