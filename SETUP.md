# Setup

## Requirements

- Python 3.11-3.13
- pip

## Local offline mode

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
~~~

Defaults:

~~~env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hash
COMMUNITY_ALGORITHM=louvain
~~~

This mode runs Local, Global, DRIFT, and Basic query control flow without paid API calls.

## Optional Leiden backend

~~~bash
pip install -r requirements-leiden.txt
~~~

Then set:

~~~env
COMMUNITY_ALGORITHM=leiden
~~~

## Real model providers

~~~env
LLM_PROVIDER=openai
LLM_API_KEY=...
LLM_MODEL=gpt-4.1-mini

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=256
~~~

## DRIFT tuning

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

Increasing depth or branch count can increase both cost and latency. DRIFT_MAX_ACTIONS is a hard safety bound on exploration work.

## Validate

~~~bash
pytest -q
python -m compileall -q .
ruff check .
~~~

## Migration

Older V1/V2/V2.1 JSON indexes remain readable because V2.2 does not require new persisted fields for DRIFT. Re-indexing is still recommended when migrating from versions that predate hierarchical communities or persisted retrieval embeddings.
