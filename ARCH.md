# graphRagX Architecture

The repository now uses the V2 architecture described in [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md).

The key invariant is that an index is not just a graph. It is a portable retrieval package containing graph structure, community reports, source text units, and the embeddings needed by each query mode.

```text
Indexing
Documents -> Chunks -> Extraction -> Directed Graph -> Communities -> Reports -> Retrieval Embeddings -> Storage

Query
Local  -> entity retrieval -> graph expansion -> source evidence -> generation
Global -> community retrieval -> report synthesis -> generation
Basic  -> text-unit retrieval -> generation
```

Core boundaries:

- `core/`: deterministic domain algorithms
- `services/`: provider/storage adapters
- `workflows/`: LangGraph orchestration and error routing
- `models/`: validated API/index contracts and graph model
- `prompts/`: prompt construction only
- `api/`: HTTP validation and response shaping

See the V2 architecture document for design decisions and the roadmap for planned production features.
