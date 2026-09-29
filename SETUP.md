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

The default configuration uses deterministic offline providers:

~~~env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hash
COMMUNITY_ALGORITHM=louvain
~~~

## Optional Leiden backend

~~~bash
pip install -r requirements-leiden.txt
~~~

Then set:

~~~env
COMMUNITY_ALGORITHM=leiden
~~~

The default Louvain backend remains available without native graph dependencies.

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

Optional compatible endpoints can be configured with LLM_BASE_URL and EMBEDDING_BASE_URL.

## Global Search tuning

~~~env
GLOBAL_COMMUNITY_LEVEL=1
GLOBAL_TOP_K=24
GLOBAL_MAX_DATA_TOKENS=8000
GLOBAL_MAP_BATCH_TOKENS=2000
GLOBAL_REDUCE_DATA_TOKENS=4000
~~~

Higher community levels are finer-grained. Query requests can override the default with community_level.

## Validate the repository

~~~bash
pytest -q
python -m compileall -q .
ruff check .
~~~

## Migration

V2 and V1 JSON indexes remain readable, but existing indexes do not contain the V2.1 hierarchy/report structure. Re-indexing is recommended after upgrading.
