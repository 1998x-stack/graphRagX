# graphRagX V2.1 Architecture

## Goal

V2 established correctness, deterministic offline execution, provenance-preserving Local Search, portable index storage, and testable provider boundaries.

V2.1 adds the next GraphRAG fidelity layer:

1. broad-to-fine hierarchical communities
2. bottom-up community reports
3. hierarchy-level-aware Global Search
4. tokenizer-aware context budgets
5. map/reduce Global Search with rated intermediate evidence

## Indexing dataflow

~~~text
Documents
  -> Text Units
  -> Entity / Relation Extraction
  -> Directed Typed Graph
  -> Weighted Undirected Projection
  -> Hierarchical Community Detection
  -> Bottom-Up Community Reports
  -> Entity / Chunk / Community Embeddings
  -> Portable Index
~~~

### Community hierarchy

Level 0 is the broadest partition. A community is recursively repartitioned when it exceeds the configured maximum cluster size. Resolution increases with depth, encouraging finer partitions.

The default backend is NetworkX Louvain. The optional Leiden backend uses python-igraph and can be enabled without changing the index/query contracts.

Every child stores parent_id. Entities can therefore appear in multiple community records across levels, but only once within each level's partition path.

### Bottom-up reports

Reports are generated deepest-first. When a parent report is generated, already-created child reports are passed into its prompt alongside the parent's entities and internal relationships.

This produces hierarchical summaries instead of independent same-level summaries.

## Global Search

~~~text
User query
   |
   v
Choose community level
   |
   v
Rank reports by query embedding
   |
   v
Apply total token budget
   |
   v
Split into map token batches
   |
   +--> Map batch 1 -> rated evidence points
   +--> Map batch 2 -> rated evidence points
   +--> ...
   |
   v
Merge + filter + sort points
   |
   v
Apply reduce token budget
   |
   v
Reduce synthesis
~~~

Map output is structured JSON. Each evidence point has a 0-100 importance score and retains the community report IDs that produced it.

The reduce stage receives only evidence points that fit the configured reduce budget.

## Local Search improvements

Local Search still starts from semantically ranked entities and graph expansion, but source text units are now reranked against the query before context packing. Token limits are enforced with the configured model tokenizer rather than by character count.

## Token budgeting

tiktoken is used for:

- Local source context
- total Global report context
- each Global map batch
- Global reduce evidence

When the configured model name is not recognized, graphRagX falls back to the o200k_base encoding.

## Failure model

Indexing and query workflows retain V2's fail-fast behavior. Errors do not silently produce partially valid indexes or continue through downstream LangGraph nodes.

Community report generation now raises on empty/failed summaries rather than persisting a failure string as if it were valid report content.

## Compatibility

Index schema version is 2.1.

V2 index payloads remain loadable because the storage contract is additive. Re-indexing is recommended to obtain hierarchical communities and bottom-up reports.

## Deferred to V2.2+

- DRIFT query state and iterative follow-up generation
- entity/relationship description summarization across repeated mentions
- graph pruning policies
- claims/covariates
- incremental/content-addressed indexing
- DuckDB/Parquet and vector-store adapters
- benchmark-driven quality/cost evaluation
