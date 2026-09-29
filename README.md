# graphRagX

graphRagX is a compact GraphRAG research and engineering baseline built around FastAPI, LangGraph, NetworkX, and pluggable LLM/embedding providers.

V2 focuses on one goal: make the repository **deterministic, testable, evidence-preserving, and easy to evolve** instead of leaving important behavior behind TODOs and random mock vectors.

## What V2 changes

- deterministic offline defaults: `mock` LLM + feature-hash embeddings
- real provider selection from environment variables
- embeddings built once at index time and persisted with the index
- Local Search uses both graph structure and source text units
- Global Search ranks community reports before synthesis
- Basic Search provides a conventional vector-RAG baseline
- directed typed relationships via `MultiDiGraph`
- duplicate relationships are merged instead of silently duplicated
- workflow failures short-circuit through LangGraph conditional edges
- safe `index_id` validation blocks path traversal
- atomic JSON writes reduce corrupted-index risk
- input/resource limits, safer CORS defaults, and opt-in LLM payload logging
- unit tests + GitHub Actions CI for Python 3.11-3.13

## Architecture

```text
Documents
   │
   ▼
Chunking ─────────────► Text Units ───────────────► Chunk Embeddings
   │
   ▼
Entity / Relation Extraction
   │
   ▼
Directed Multi-Relation Knowledge Graph
   │
   ├────────► Entity Embeddings
   │
   └────────► Louvain Communities ─► Community Reports ─► Community Embeddings

Persisted index
= graph + communities + text units + retrieval embeddings + metadata
```

Query modes:

```text
local  : query -> entity seeds -> graph expansion -> source text units -> answer
global : query -> ranked community reports -> synthesis -> answer
basic  : query -> ranked text units -> answer
```

See [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) for design details and [`docs/ROADMAP.md`](docs/ROADMAP.md) for the next stages.

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install

```bash
pip install -r requirements.txt
cp .env.example .env
```

The default `.env` is fully offline and does not make paid API calls.

### 3. Run

```bash
python main.py
```

Open:

- API docs: `http://localhost:8000/docs`
- health: `http://localhost:8000/api/v1/health`

## Provider modes

Safe defaults:

```env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hash
```

To use OpenAI-compatible providers:

```env
LLM_PROVIDER=openai
LLM_API_KEY=...
LLM_MODEL=gpt-4.1-mini

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=256
```

`hash` embeddings are deterministic lexical feature hashing. They are useful for tests and offline development, but they are **not** a replacement for a semantic embedding model in production.

## API examples

### Create an index

```bash
curl -X POST http://localhost:8000/api/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "index_id": "demo",
    "documents": [
      "Alice works at Acme on graph retrieval systems.",
      "Acme develops retrieval and knowledge graph software."
    ]
  }'
```

Existing index IDs are protected from accidental replacement. To replace one explicitly:

```json
{"index_id":"demo","documents":["..."],"overwrite":true}
```

### Local GraphRAG

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What does Alice work on?","mode":"local","top_k":5}'
```

### Global GraphRAG

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What are the main themes?","mode":"global","top_k":5}'
```

### Basic vector RAG

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What does Acme build?","mode":"basic","top_k":5}'
```

### Inspect / delete indexes

```bash
curl http://localhost:8000/api/v1/indexes
curl http://localhost:8000/api/v1/query/stats/demo
curl -X DELETE http://localhost:8000/api/v1/index/demo
```

## Index lifecycle

The V2 index payload stores:

- entities and typed relations
- communities and community reports
- source text units
- entity embeddings
- text-unit embeddings
- community-report embeddings
- provider/model/dimension metadata
- source metadata

When the configured embedding provider/model/dimension differs from the stored index metadata, query logic ignores incompatible stored vectors and rebuilds candidate vectors using the current provider. This preserves correctness at the cost of query-time work.

## Safety and privacy defaults

- `index_id` accepts only letters, numbers, `.`, `_`, and `-` and cannot contain path separators.
- graph writes are atomic (`tempfile` + `os.replace`).
- raw prompts and model responses are **not** persisted unless `STORE_LLM_PAYLOADS=true`.
- CORS defaults to localhost development origins instead of wildcard credentials.
- document count, payload size, query length, and `top_k` are bounded.

These controls are a baseline, not a substitute for authentication, authorization, rate limiting, tenant isolation, or production secret management.

## Development

```bash
pytest -q
python -m compileall -q .
ruff check .
```

CI runs compilation checks and the test suite across Python 3.11, 3.12, and 3.13; Ruff remains an optional local quality check.

## Current limitations

V2 is intentionally a foundation, not a full clone of Microsoft GraphRAG. The next fidelity improvements are hierarchical Leiden communities, map/reduce Global Search, DRIFT, incremental indexing, richer entity resolution, token-budgeted context packing, real vector-store adapters, and evaluation datasets.

## References

- Microsoft GraphRAG: https://microsoft.github.io/graphrag/
- Query overview: https://microsoft.github.io/graphrag/query/overview/
- Local Search: https://microsoft.github.io/graphrag/query/local_search/
- Global Search: https://microsoft.github.io/graphrag/query/global_search/
- DRIFT Search: https://microsoft.github.io/graphrag/query/drift_search/
- LangGraph `StateGraph`: https://reference.langchain.com/python/langgraph/graph/state/StateGraph

## License

MIT
