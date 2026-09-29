# Setup

## Requirements

- Python 3.11-3.13
- pip

## Local offline mode

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

The default configuration uses deterministic offline providers:

```env
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hash
```

This mode is suitable for workflow tests, API development, and reproducible examples.

## Real model providers

Set the provider and credentials in `.env`:

```env
LLM_PROVIDER=openai
LLM_API_KEY=...
LLM_MODEL=gpt-4.1-mini

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=256
```

Optional custom compatible endpoints can be configured with `LLM_BASE_URL` and `EMBEDDING_BASE_URL`.

## Validate the repository

```bash
pytest -q
python -m compileall -q .
ruff check .
```

## Important migration note from V1

V2 can read the original flat graph JSON format, but old indexes do not contain text units or persisted embeddings. They remain queryable; missing or incompatible vectors are generated on demand. Re-indexing is recommended to get the full V2 retrieval path and provenance.
