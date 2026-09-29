# graphRagX V2.3 — Graph Quality & Retrieval Quality

## Goals

V2.3 improves the quality of the graph before community detection and makes query
quality measurable across Basic, Local, Global, and DRIFT.

## Indexing pipeline

~~~text
documents
  -> chunks
  -> entity/relation extraction
  -> conservative entity resolution
  -> graph merge
  -> multi-mention description summarization
  -> graph pruning
  -> optional claim/covariate extraction
  -> hierarchical communities
  -> bottom-up reports
  -> retrieval embeddings
  -> persisted quality report
~~~

## Entity resolution

The default strategy is `normalized_exact`.

Normalization uses Unicode NFKC, case-folding, punctuation removal, and whitespace
collapse. Merges are constrained by entity type. No fuzzy matching is performed.

The resolver rewrites relationship endpoints to canonical entity names and preserves
all observed aliases.

This is intentionally conservative: a false entity merge can contaminate graph
neighborhoods, community structure, claims, and every downstream query mode.

## Mention preservation and description summaries

Entities and relationships now persist:

- `mention_count`
- raw `description_mentions`
- entity `aliases`

When repeated descriptions exist, an OpenAI-backed runtime can summarize them into a
single retrieval description while retaining the original mentions. Offline/mock mode
uses deterministic deduplicated concatenation.

## Graph pruning

The pruner is a separate indexing stage and records both pre/post statistics.

Supported controls:

- minimum node frequency
- maximum node-frequency standard-deviation threshold
- minimum node degree
- maximum node-degree standard-deviation threshold
- minimum edge-weight percentile
- optional ego-node removal
- optional largest-connected-component restriction

Default thresholds are intentionally neutral.

The persisted quality report records removed node counts, removed relationship counts,
the effective edge threshold, and per-node removal reasons.

## Claims / covariates

Claim extraction is optional and disabled by default.

A claim stores:

- type
- description
- canonical subject entity
- optional object entity
- status
- optional date range
- source text
- source text-unit ID

Claims are stored as covariates, not graph edges. Local and DRIFT retrieval expose
claims connected to the relevant entity neighborhood as an additional evidence surface.

## Quality observability

Each V2.3 index persists a `quality_report` with reports from:

- entity resolution
- description summarization
- graph pruning
- claim extraction

API:

~~~text
GET /api/v1/index/{index_id}/quality
~~~

Query stats also report the number of claims and the index quality report.

## Evaluation harness

Evaluation cases are JSONL records with:

- query
- index ID
- modes to execute
- expected source IDs
- expected answer terms
- retrieval parameters

Run:

~~~bash
python -m scripts.evaluate \
  --dataset evaluation/sample.jsonl \
  --output outputs/evaluation/report.json
~~~

Deterministic metrics:

- source recall
- answer-term coverage
- latency
- success rate
- DRIFT action count

No LLM judge is required, making the baseline reproducible and suitable for CI/regression
work. External judge metrics can be layered on later but should not replace deterministic
retrieval metrics.

## Alignment with Microsoft GraphRAG

Current GraphRAG exposes separate configuration/workflow concepts for description
summarization, manual graph pruning, and optional claim extraction. Claim extraction is
off by default because useful claim prompts are domain dependent.

References:

- https://microsoft.github.io/graphrag/config/yaml/
- https://microsoft.github.io/graphrag/index/default_dataflow/
- https://microsoft.github.io/graphrag/index/outputs/
