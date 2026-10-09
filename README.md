# graphRagX

graphRagX is a compact GraphRAG research and engineering baseline built with FastAPI, LangGraph, NetworkX, tokenizer-aware context budgeting, and pluggable LLM/embedding/community backends.

V2.3 adds graph-quality augmentation and keeps four query modes:

- Local Search
- Global Search
- DRIFT Search
- Basic vector RAG

## Architecture

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
           +--> bottom-up community reports
           +--> community-report embeddings

Query paths

Local:
query -> entity seeds -> graph expansion -> source evidence -> answer

Global:
query -> community reports -> map batches -> rated evidence -> reduce

DRIFT:
query -> community primer -> ranked follow-ups
      -> iterative local retrieval -> confidence/provenance ledger
      -> bounded traversal -> reduce

Basic:
query -> text-unit vectors -> answer
~~~

## V2.3 Graph Quality & Retrieval Quality

V2.3 adds quality stages before community detection:

~~~text
Extract
  -> Entity resolution
  -> Graph merge
  -> Entity/relation description consolidation
  -> Graph pruning
  -> Optional claims/covariates
  -> Communities + retrieval indexes
~~~

Key properties:

- conservative normalized-exact alias resolution
- entity and relationship mention counts
- preserved raw description mentions
- auditable graph pruning with before/after statistics
- optional claim extraction, disabled by default
- claim-aware Local and DRIFT context
- per-index quality reports
- deterministic four-mode evaluation harness

Inspect index quality:

~~~bash
curl http://localhost:8000/api/v1/index/demo/quality
~~~

Run a cross-mode evaluation:

~~~bash
python -m scripts.evaluate \
  --dataset evaluation/sample.jsonl \
  --output outputs/evaluation/report.json
~~~

The baseline evaluator reports source recall, answer-term coverage, latency, success rate,
and DRIFT action count without requiring an LLM judge.

## V2.2 DRIFT

DRIFT starts broader than Local Search by priming from relevant community reports, then turns uncertainty into specific follow-up questions and investigates them with local graph retrieval.

The runtime keeps an explicit trace containing:

- primer answer and community report IDs
- follow-up question hierarchy
- intermediate answers
- confidence
- source IDs
- traversal depth
- action count
- termination reason

Traversal is bounded by depth, branch count, total actions, token budgets, confidence gating, and duplicate-question suppression.

Example:

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "index_id": "demo",
    "query": "How do the major themes connect to specific evidence?",
    "mode": "drift",
    "community_level": 1
  }'
~~~

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

API docs: http://localhost:8000/docs

The default configuration is offline:

~~~env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hash
COMMUNITY_ALGORITHM=louvain
~~~

For Leiden:

~~~bash
pip install -r requirements-leiden.txt
~~~

~~~env
COMMUNITY_ALGORITHM=leiden
~~~

## Query examples

Local:

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What does Alice work on?","mode":"local","top_k":5}'
~~~

Global:

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What are the main themes?","mode":"global","top_k":20,"community_level":1}'
~~~

DRIFT:

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"How do global themes connect to local facts?","mode":"drift","community_level":1}'
~~~

Basic:

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What does Acme build?","mode":"basic","top_k":5}'
~~~

## Important DRIFT settings

~~~env
DRIFT_K_FOLLOWUPS=3
DRIFT_PRIMER_FOLDS=3
DRIFT_N_DEPTH=2
DRIFT_MAX_ACTIONS=12
DRIFT_COMMUNITY_LEVEL=1
DRIFT_PRIMER_DATA_TOKENS=5000
DRIFT_LOCAL_MAX_DATA_TOKENS=2500
DRIFT_REDUCE_DATA_TOKENS=5000
DRIFT_LOCAL_TOP_K_ENTITIES=5
DRIFT_EXPANSION_MIN_CONFIDENCE=0.35
~~~

## Development

~~~bash
pytest -q
python -m compileall -q .
ruff check .
~~~

CI validates Python 3.11, 3.12, and 3.13. A separate job installs python-igraph and smoke-tests the optional Leiden backend.

## Architecture docs

- docs/ARCHITECTURE_V2.md
- docs/ARCHITECTURE_V2_1.md
- docs/ARCHITECTURE_V2_2.md
- docs/ARCHITECTURE_V2_3.md
- docs/ROADMAP.md

## References

- Microsoft GraphRAG: https://microsoft.github.io/graphrag/
- Local Search: https://microsoft.github.io/graphrag/query/local_search/
- Global Search: https://microsoft.github.io/graphrag/query/global_search/
- DRIFT Search: https://microsoft.github.io/graphrag/query/drift_search/
- Configuration: https://microsoft.github.io/graphrag/config/yaml/
- LangGraph StateGraph: https://reference.langchain.com/python/langgraph/graph/state/StateGraph

## License

MIT
