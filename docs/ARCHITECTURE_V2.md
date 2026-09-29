# graphRagX V2 Architecture

## 1. Why V2 exists

The original repository had a useful directory skeleton, but the runtime behavior was still a prototype: the default embedding implementation generated random vectors, every local query re-embedded the whole entity set, the LLM provider was hard-coded to a mock, workflow failures continued into downstream nodes, graph edges lost relationship direction, and an arbitrary `index_id` could become part of a filesystem path.

V2 turns the repository into a deterministic, testable GraphRAG baseline while keeping the code small enough for learning and experimentation.

## 2. Architecture

```text
Documents
   │
   ▼
Chunking ──► TextUnits ───────────────────────────────┐
   │                                                  │
   ▼                                                  ▼
Entity/Relation Extraction                     Chunk Embeddings
   │
   ▼
Directed Typed Knowledge Graph
   │
   ├──► Louvain Communities ─► Community Reports ─► Community Embeddings
   │
   └──► Entity Embeddings

Persisted Index = graph + text units + reports + retrieval embeddings + metadata
```

Query paths:

```text
Local  : query embedding -> entity seeds -> graph expansion -> source text units -> answer
Global : query embedding -> ranked community reports -> synthesis -> answer
Basic  : query embedding -> ranked source text units -> answer
```

The Basic mode is intentionally included as a control. A GraphRAG implementation should be compared with a conventional vector-RAG path rather than assuming the graph is always superior.

## 3. Key design decisions

### Deterministic offline defaults

`LLM_PROVIDER=mock` and `EMBEDDING_PROVIDER=hash` are safe defaults. The hash embedding is a deterministic lexical feature-hashing baseline, not a semantic model. It exists so tests and examples are reproducible without credentials or network access.

### Build retrieval artifacts at index time

Entity, text-unit, and community vectors are created once during indexing and persisted with the graph. Query-time work is reduced to one query embedding plus vector comparisons.

### Evidence-preserving local search

Local search retrieves entity seeds, expands the graph, then resolves `source_chunk_ids` back to raw text units. The final context therefore contains both graph structure and source evidence.

### Typed, directed graph

A `MultiDiGraph` preserves relationship direction and multiple relation types between the same pair of entities. Community detection operates on a weighted undirected projection of that graph.

### Explicit failure routing

Workflows use modern `StateGraph` and conditional edges. A failed node routes directly to `END`; downstream nodes never execute against incomplete state.

### Storage boundary

The current adapter is local JSON for simplicity, but the persisted `GraphData` is now a portable index contract. This creates a clean migration path to SQLite/DuckDB/Parquet plus a vector database without changing query semantics.

## 4. Production evolution path

The next layer should add: true Leiden hierarchical communities; map/reduce Global Search; DRIFT-style exploration; incremental indexing with content hashes; entity resolution beyond exact names; tokenizer-aware context budgets; a real vector-store adapter; tracing/metrics; authentication and tenant isolation; and benchmark datasets with retrieval/answer quality metrics.

## 5. External design references

- Microsoft GraphRAG query overview: https://microsoft.github.io/graphrag/query/overview/
- Microsoft GraphRAG local search: https://microsoft.github.io/graphrag/query/local_search/
- Microsoft GraphRAG global search: https://microsoft.github.io/graphrag/query/global_search/
- Microsoft GraphRAG DRIFT search: https://microsoft.github.io/graphrag/query/drift_search/
- LangGraph StateGraph reference: https://reference.langchain.com/python/langgraph/graph/state/StateGraph
