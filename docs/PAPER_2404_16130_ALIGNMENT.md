# GraphRAG paper fidelity and engineering acceptance

> Research basis: Darren Edge et al., **From Local to Global: A GraphRAG Approach to Query-Focused Summarization**, arXiv:2404.16130.
>
> Comparison baseline: graphRagX `main` at `43ad338d7ecaabf4df019044e0ce832268944847`.
>
> This document distinguishes **implemented**, **partial**, and **not yet implemented**. It is not a claim that graphRagX reproduces the paper's published numbers.

## 1. Reproducible methodology map

| Paper section | Contract / target | Repository module | Fidelity status |
|---|---|---|---|
| 3.1.1 | Source document -> token-size overlapping text units | `core/chunking.py` | **Implemented optionally**: `CHUNK_UNIT=tokens`, source offsets and paper-style 600/100 preset; default characters mode remains for compatibility |
| 3.1.2 | LLM entity and relation extraction, descriptions, strict validation | `core/extraction.py`, `prompts/extraction_prompts.py` | **Implemented baseline + stronger failure semantics**: bounded retry and fail-closed invalid responses; domain prompt tuning remains manual |
| 3.1.3 | Aggregate repeated entity and relation mentions into weighted graph | `core/entity_resolution.py`, `models/graph.py`, `core/description_summarization.py` | **Partial**: normalized exact resolution and aggregation; alias ambiguity and extraction recall remain to be benchmarked |
| 3.1.4 | Hierarchical Leiden communities, complete graph coverage at C0-C3 | `core/community.py` | **Implemented complete cut projection**; optional Leiden, but Louvain is default; backend equivalence not demonstrated |
| 3.1.5 | Bottom-up community reports, node/edge/claim inclusion, importance ordering | `core/summarization.py`, `prompts/summary_prompts.py` | **Partial**: bottom-up reports exist; claim-aware report context and graph prominence packing still required |
| 3.1.6 | Shuffle all community reports, map with helpfulness score, reduce | `workflows/query_workflow.py` | **Improved**: global coverage is default, stable shuffle, budget-fair per-report packing, existing parallel map/reduce; per-report token truncation remains a quality tradeoff |
| 3.2 | Persona -> task -> global question generation | Future `evaluation/question_generation.py` | **Not implemented**: questions must be curated externally |
| 3.3 | Blind pairwise judge, repeats, 4 criteria, tie handling | `evaluation/paper_judge.py`, `scripts/paper_evaluate.py` | **Implemented evaluation mechanism**, not paper replication; model judge calibration, uncertainty estimation and factuality checks still required |
| 4.1 | C0/C1/C2/C3 vs text-map-reduce (TS) vs semantic search (SS) on matched data | `evaluation/paper_conditions.py`, `scripts/paper_conditions.py` | **Implemented six-arm orchestration**; corpus, model, prompts and budget parity still unverified; not a published-results replication |
| 4.2 | Claim-based comprehensiveness and diversity analysis | `core/claims.py` | **Not implemented** for answer-level Claimify-style analysis; index-time covariates are not the same thing |

The four modes in the repository serve different tasks:
- **Basic** = standard top-k text-unit embedding retrieval, closest to the paper's SS baseline.
- **Global** = community-based query-focused map/reduce, the central paper method.
- **Text Summary** (`text_summary`) = shuffled source-text map/reduce; the paper's TS control arm (subject to explicit budget limits).
- **Local** = entity/neighbor/source exploration; not the paper's core global experiment.
- **DRIFT** = later hybrid iterative search, also not one of the original six comparison conditions.

## 2. Correctness contracts introduced in this branch

### Hierarchical partition and coverage

- A partition returned by Louvain/Leiden must contain each input node exactly once; duplicate or missing nodes fail indexing.
- A requested level yields a *cut* of the hierarchy, not only communities with that exact level.
- If a parent has no descendants at the requested depth, the parent is projected into the cut.
- The community cut therefore preserves the coverage of unrefined branches. This is the paper's C0-C3 projection semantics.
- The child partition is computed once and reused, avoiding a second potentially expensive/different partition call.

### Global report coverage

- `GLOBAL_SELECTION_MODE=coverage` is the paper-oriented default. `top_k` does **not** silently discard entire communities in this mode.
- Reports are sorted and then deterministically shuffled using a SHA-256-derived per-question seed. This reproduces *the shuffle design*, not the paper's random draws.
- A tokenizer-aware allocation distributes the index-level report budget fairly. Every report retains at least the configured minimum token allowance, or the query fails explicitly.
- `GLOBAL_MAX_REPORTS` protects against unexpectedly expensive corpus-wide queries; exceeding it is an error, not a partial answer presented as complete.
- `TS_MAX_TEXT_UNITS` and `TS_MAX_DATA_TOKENS` provide the same explicit all-source coverage controls for the paper's text-summary control arm.
- Metadata includes `community_cut_size`, `community_reports_included`, and `community_report_coverage`.
- `GLOBAL_SELECTION_MODE=relevance` preserves the existing retrieval-first behavior for lower-cost operation and DRIFT primer retrieval.

**Tradeoff:** A full-coverage budget may truncate individual reports. Increasing `GLOBAL_MAX_DATA_TOKENS` improves report fidelity but raises map/reduce costs. A later iteration should support streaming report batches without a corpus-level cap, with explicit cost budgets and resumable map tasks.

### Extraction failures

- Invalid JSON, malformed schemas, missing relation endpoints, Pydantic failures and provider exceptions are retried within a fixed bound.
- After retries are exhausted, indexing fails instead of persisting a misleading graph built from silently empty extractions.
- A valid structured response with `entities=[]` and `relations=[]` is still accepted.

## 3. How to verify

```bash
pip install -r requirements.txt
python -m compileall -q .
pytest -q
pytest -q tests/test_paper_fidelity.py tests/test_extraction_fail_closed.py tests/test_paper_judge.py
```

Generate one index with the repository's existing `/api/v1/index` endpoint. The default mock LLM demonstrates the pipeline but does **not** demonstrate factual extraction or summarization quality.

A corpus-wide Global query:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","mode":"global","query":"What themes recur across the entire collection?","community_level":1,"top_k":5}'
```

To compare research questions consistently:

```bash
python -m scripts.paper_conditions \
  --dataset evaluation/sample.jsonl \
  --output outputs/evaluation/answers.json --max-cases 5

# Only after configuring a real LLM evaluator:
LLM_PROVIDER=openai python -m scripts.paper_evaluate \
  --input outputs/evaluation/answers.json \
  --output outputs/evaluation/pairwise.json \
  --mode-a C2 --mode-b SS --repeats 3
```

`evaluation/sample.jsonl` is a format/example fixture, **not** the paper's benchmark dataset. For meaningful results, supply a corpus-matched question set, same model and prompt budget per mode, deterministic index snapshot, relevant costs, and replicated judge prompts. The judge requires an actual model; offline mock scores are intentionally prohibited.

## 4. Next engineering stages (explicit unimplemented scope)

### Stage A: Paper-faithful indexing (highest priority)

1. **Token-aware source ingestion hardening**: optional `CHUNK_UNIT=tokens`, exact source offsets and 600/100 preset are implemented; next add source document hash lineage, sentence-aware boundary ablations and tokenizer-version manifests.
2. **Extract evidence grounding**: store exact text spans per entity, relation and claim; validate that source spans exist before output.
3. **Entity fusion**: type-compatible aliases with merge audit/confidence and reversal; benchmark precision/recall against manually annotated corpora.
4. **Leiden fidelity**: make Leiden a research preset, verify seed/version stability, modularity and coverage, and test disconnected components.
5. **Community reports**: edge-degree-based prominence ordering, include claims, explicit source IDs; budget-aware child report substitution rather than unrestricted hard slicing.
6. **Index manifests**: record input content hashes, model/provider/version, tokenizer, prompts, algorithm parameters, date, schema version, embedding dimensions, and source-to-graph lineage.

**Exit criteria:** a fixed corpus produces a deterministic schema-valid index with complete community cuts, preserved source coverage, model-call accounting, and an audit trail back to text spans.

### Stage B: Retrieval quality and factuality

1. Local seed entity hybrid BM25/vector retrieval; relation-aware reranking and controlled neighborhood selection.
2. Global map provenance per point, map-point to report references, and report to original source chunks.
3. Citation verification (source existence, supported claim, no invented citations), abstention and contradiction handling.
4. Query planner to distinguish local fact lookup from corpus-wide sensemaking; never route only by lexical keyword rules.
5. Incremental update/delete and cache invalidation for graph, reports and embeddings.
6. Provider-agnostic embedding validation and batch/vector-store adapters.

**Exit criteria:** source-grounding precision and recall improve on held-out questions without regression in 4 query modes.

### Stage C: Original paper evaluation and ablations

1. Validate the implemented six-condition **C0 / C1 / C2 / C3 / TS / SS** runner on a frozen corpus, ensuring matched prompts, model budgets, index provenance and scoring.
2. Generate persona/task/question sets using the 3.2 method and manually audit global vs local question leakage.
3. Run independent answer generation and blinded comparison for comprehensiveness, diversity, empowerment and directness.
4. Use repeated balanced A/B order, per-question aggregated scores and confidence intervals with a question-level cluster bootstrap.
5. Report answer factuality with verifiable answer-level claims (not index-time claims), redundancy and source support.
6. Ablate chunk size/overlap, prompts, entity resolution, Leiden vs Louvain, report levels, map budgets, score thresholds and model choice.
7. Compare index latency/cost, query latency/cost, tokens, source recall, answer completeness, diversity and trace failures.

**Exit criteria:** reproducible datasets, frozen index IDs, evaluation scripts, run metadata, plots, and honest differences from the paper's published dataset/models.

### Stage D: Production and operations

1. Resume/checkpoint indexing, idempotency keys, transactional publish, replay and cancel.
2. Durable object-store + Parquet/DuckDB/vector-store adapters, index version migrations and integrity verification.
3. Document ingestion for PDF/DOCX/HTML with metadata and ACL-aware permissions; test prompt-injection containment.
4. Authentication, tenant isolation, quotas, API pagination, rate limiting and retention policies.
5. OpenTelemetry traces for extraction, embeddings, community graph, map/reduce and prompt/token costs.
6. CI performance baselines, security/dependency review, Docker deployment, load/stress testing and reproducible releases.

**Exit criteria:** resumability and security tests, per-tenant isolation, measurable SLOs and rollback-tested index publication.

## 5. Honest limitations

- No empirical claim is made about improving RAG answer quality until held-out tests are executed.
- Source-aware citation verification and true scientific replication remain outstanding.
- The paper is about **global query-focused summarization**, not proof that graph retrieval universally beats vector search.
- Real graph indexing and LLM pairwise judging can be expensive; run a small fixture before scaling.
- This branch changes behavior under Global coverage mode. Existing users needing previous top-k semantics should choose `GLOBAL_SELECTION_MODE=relevance`.

## Sources

- Paper: https://arxiv.org/abs/2404.16130
- Paper HTML: https://ar5iv.labs.arxiv.org/html/2404.16130
- Microsoft GraphRAG: https://github.com/microsoft/graphrag
- Project-specific baseline: https://github.com/1998x-stack/graphRagX
