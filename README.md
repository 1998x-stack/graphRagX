# graphRagX

graphRagX is a compact GraphRAG research and engineering baseline built with FastAPI, LangGraph, NetworkX, tokenizer-aware context budgeting, and pluggable LLM/embedding/community backends.

V2.1 focuses on GraphRAG fidelity while preserving the deterministic and testable V2 foundation.

## What is implemented

- deterministic offline defaults: mock LLM + feature-hash embeddings
- configurable OpenAI LLM and embedding providers
- index-time entity, text-unit, and community-report embeddings
- directed typed relationships via MultiDiGraph
- hierarchical broad-to-fine communities
- Louvain by default, optional Leiden backend through python-igraph
- bottom-up community report generation
- Local Search with entity seeds, graph expansion, provenance-ranked source text units, and token budgets
- Global Search with hierarchy-level selection, report ranking, token-budgeted map batches, rated intermediate evidence, and reduce synthesis
- Basic Search as a conventional vector-RAG baseline
- workflow error short-circuiting
- atomic index writes and path-safe index IDs
- tests and CI across Python 3.11, 3.12, and 3.13

## V2.1 architecture

~~~text
Documents
   |
   v
Text Units -------------------------------> Text-Unit Embeddings
   |
   v
Entity / Relation Extraction
   |
   v
Directed Multi-Relation Knowledge Graph
   |
   +--> Entity Embeddings
   |
   +--> Hierarchical Communities
           |
           +--> fine reports
           +--> parent reports built bottom-up
           +--> community-report embeddings

Global query:
query -> hierarchy level -> ranked reports -> token batches
      -> parallel map -> rated evidence points -> filter/rank
      -> reduce -> final answer
~~~

The default community backend is Louvain because it has no extra native dependency. To use Leiden:

~~~bash
pip install -r requirements-leiden.txt
~~~

Then configure:

~~~env
COMMUNITY_ALGORITHM=leiden
~~~

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
~~~

Windows PowerShell activation:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

API docs: http://localhost:8000/docs

## Query modes

### Local

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What does Alice work on?","mode":"local","top_k":5}'
~~~

Local Search ranks entity seeds, expands graph neighbors, resolves source chunk provenance, reranks source chunks against the query, and packs them under LOCAL_MAX_DATA_TOKENS.

### Global

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What are the main themes?","mode":"global","top_k":20,"community_level":1}'
~~~

Global Search uses reports from one hierarchy level, ranks them against the query, caps total context with GLOBAL_MAX_DATA_TOKENS, batches map calls with GLOBAL_MAP_BATCH_TOKENS, then reduces ranked evidence under GLOBAL_REDUCE_DATA_TOKENS.

If the requested community level is unavailable, graphRagX selects the nearest available lower level.

### Basic

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What does Acme build?","mode":"basic","top_k":5}'
~~~

Basic Search is the control path for evaluating whether graph structure adds value over normal vector retrieval.

## Community hierarchy

Important settings:

~~~env
COMMUNITY_ALGORITHM=louvain
COMMUNITY_MAX_LEVELS=3
COMMUNITY_MAX_CLUSTER_SIZE=20
COMMUNITY_RESOLUTION_MULTIPLIER=1.5
LEIDEN_RESOLUTION=1.0
LEIDEN_SEED=42
~~~

Level 0 is broadest. Higher levels are finer. Communities larger than COMMUNITY_MAX_CLUSTER_SIZE are recursively partitioned until they are small enough, cannot be split further, or COMMUNITY_MAX_LEVELS is reached.

Reports are generated from the deepest level upward, so parent reports can use already-generated child reports.

## Provider modes

Safe defaults:

~~~env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hash
~~~

Real providers:

~~~env
LLM_PROVIDER=openai
LLM_API_KEY=...
LLM_MODEL=gpt-4.1-mini

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=256
~~~

Hash embeddings are deterministic lexical feature hashing for tests and offline development; they are not a semantic embedding substitute for production.

## Development

~~~bash
pytest -q
python -m compileall -q .
ruff check .
~~~

## Current next steps

V2.1 intentionally stops before implementing a full DRIFT engine. The next priority is DRIFT-style local/global traversal, followed by incremental indexing, stronger entity resolution, graph pruning, vector-store adapters, and evaluation datasets comparing Basic / Local / Global / DRIFT.

See docs/ARCHITECTURE_V2_1.md and docs/ROADMAP.md.

## References

- Microsoft GraphRAG: https://microsoft.github.io/graphrag/
- Global Search: https://microsoft.github.io/graphrag/query/global_search/
- Local Search: https://microsoft.github.io/graphrag/query/local_search/
- DRIFT Search: https://microsoft.github.io/graphrag/query/drift_search/
- GraphRAG configuration: https://microsoft.github.io/graphrag/config/yaml/
- LangGraph StateGraph: https://reference.langchain.com/python/langgraph/graph/state/StateGraph

## License

MIT
