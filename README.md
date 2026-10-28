# graphRagX

graphRagX is a compact GraphRAG research and engineering baseline built with FastAPI, LangGraph, NetworkX, tokenizer-aware context budgeting, and pluggable LLM/embedding/community backends.

GraphRagX V2.4 extends the V2.3 quality baseline with a fifth query mode and
full-coverage Global Search for corpus-level questions:

- Local Search
- Global Search
- DRIFT Search
- Basic vector RAG
- Text Summary (TS): Map-Reduce directly over source text chunks (paper control arm)

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
query -> complete hierarchy cut -> balanced report coverage
      -> shuffled map batches -> rated evidence -> reduce

Text Summary (TS):
query -> shuffled original text units -> map batches -> rated evidence -> reduce

DRIFT:
query -> community primer -> ranked follow-ups
      -> iterative local retrieval -> confidence/provenance ledger
      -> bounded traversal -> reduce

Basic:
query -> text-unit vectors -> answer
~~~

## Research alignment: arXiv 2404.16130

The branch addresses an important global-search problem: an ordinary top-k report
search does not represent the entire indexed corpus. The paper-oriented default is
`GLOBAL_SELECTION_MODE=coverage`, which maps **all** reports in the requested
complete community cut (subject to explicit resource limits), with deterministic
shuffle and token-aware budget allocation. No community is silently omitted.

```env
GLOBAL_SELECTION_MODE=coverage
GLOBAL_MAX_REPORTS=256
GLOBAL_MIN_REPORT_TOKENS=16
GLOBAL_MAX_DATA_TOKENS=8000
GLOBAL_SHUFFLE_SEED=42
TS_MAX_TEXT_UNITS=256
TS_MAX_DATA_TOKENS=8000
```

Token-based paper-style text units are supported without changing the
default character-based behavior. Set `CHUNK_UNIT=tokens`,
`CHUNK_SIZE=600`, `CHUNK_OVERLAP=100`. Chunk records retain
character and token source offsets. See `.env.paper.example` for an
experimental configuration; indexes built using different chunk units
must be treated as different experimental conditions.

The six-condition harness runs C0/C1/C2/C3 (Global at projected levels
0-3), TS (source-text map/reduce) and SS (basic semantic search):

```bash
python -m scripts.paper_conditions \
  --dataset evaluation/sample.jsonl \
  --output outputs/evaluation/paper_conditions.json --max-cases 5
```

The resulting report can be judged using
`python -m scripts.paper_evaluate --input outputs/evaluation/paper_conditions.json --mode-a C2 --mode-b SS`
with a real evaluator configured. The sample JSONL is a format fixture,
not a validated benchmark.

For the older, cheaper query-ranked behavior use
`GLOBAL_SELECTION_MODE=relevance`; in that mode `top_k` limits reports.
A query with too many reports or insufficient coverage budget fails explicitly.
The JSON query API returns `diagnostics` with selected report counts and coverage.

The new `text_summary` query mode provides the paper's TS control arm:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"index_id":"demo","query":"What themes recur?","mode":"text_summary"}'
```

Paper-style *blind pairwise* evaluation is separate from deterministic QA.
After preparing an evaluation JSONL file and a saved index:

```bash
python -m scripts.evaluate --dataset evaluation/sample.jsonl \
  --output outputs/evaluation/answers.json
LLM_PROVIDER=openai python -m scripts.paper_evaluate \
  --input outputs/evaluation/answers.json \
  --output outputs/evaluation/pairwise.json --repeats 3
```

Use a real evaluator, not the mock model, for pairwise results.
See [Paper Fidelity and Acceptance](docs/PAPER_2404_16130_ALIGNMENT.md)
for the paper-by-paper coverage table, unimplemented features, and acceptance plan.
Indexed artifacts additionally persist SHA-256 source document hashes and
chunk/model/extraction parameters for research provenance. Community reports
now prioritize high-degree entities/relationships and include optionally
extracted source-linked claims.

**The repository has not reproduced the paper's published benchmark scores.**

## Hugging Face multidimensional evaluation (V2.5 research branch)

This branch adds a **real-data evaluation harness**, not fabricated benchmark
scores. It supports version-pinned streaming datasets, one index per question,
multi-mode comparisons, source-evidence mapping, graph quality diagnostics,
robustness experiments and paired bootstrap confidence intervals.

Supported datasets: **HotpotQA** (annotated multi-hop QA), **MuSiQue**
(multi-hop decomposition), **QASPER** (scientific-paper QA), and **Multi-News**
(multi-document summaries). Multi-News is subject to *noncommercial
research/education-only terms* and requires an explicit acknowledgment.
RAGBench is reserved for future judge calibration; its existing labels
must not be applied to freshly generated answers.

```bash
pip install -r requirements.txt -r requirements-eval.txt

# Download an immutable, SHA-256-manifested validation sample.
python -m scripts.hf_eval download \
  --dataset hotpotqa --count 20 --seed 42 \
  --output data/hf/hotpotqa-val.jsonl

# Smoke test only: mock/hash outputs are NOT scientific scores.
python -m scripts.hf_eval run \
  --input data/hf/hotpotqa-val.jsonl \
  --conditions basic local global drift text_summary \
  --allow-mock-smoke \
  --output outputs/evaluation/hf-smoke.json

# Configure real LLM + embedding providers before quality comparisons.
python -m scripts.hf_eval run \
  --input data/hf/hotpotqa-val.jsonl \
  --conditions basic local global drift text_summary \
  --output outputs/evaluation/hf-real.json

python -m scripts.hf_eval compare \
  --input outputs/evaluation/hf-real.json \
  --candidate global --baseline basic --dataset hotpotqa \
  --output outputs/evaluation/hf-paired.json
```

Metrics include EM/F1, ROUGE-L, annotated source recall/precision/MRR/nDCG,
community report coverage, graph integrity/source-link coverage, failure rate,
index/query latency (including P95), and dataset/difficulty/hop-count/variant
slices. Missing gold labels are `null`, not invented scores.

For reproducible perturbations use
`--variants original shuffle distractor drop_support`.
Removing annotated gold supports does **not** establish that a question becomes
unanswerable; answer correctness is intentionally left unscored.

Create a readable report from any stored benchmark JSON:

```bash
python -m scripts.hf_report \
  --input outputs/evaluation/hf-real.json \
  --output-dir outputs/evaluation/hf-report
```

Use `--plots` with optional matplotlib installed to render per-dataset
quality figures; `summary.md` and `per_question.csv` are always generated.

Read [HF Evaluation Protocol](docs/HF_EVALUATION_PROTOCOL.md) for dataset
licenses, implementation constraints, experiment budgets, the phased study,
and next implementation gaps.

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
- docs/PAPER_2404_16130_ALIGNMENT.md

## References

- Microsoft GraphRAG: https://microsoft.github.io/graphrag/
- Local Search: https://microsoft.github.io/graphrag/query/local_search/
- Global Search: https://microsoft.github.io/graphrag/query/global_search/
- DRIFT Search: https://microsoft.github.io/graphrag/query/drift_search/
- Configuration: https://microsoft.github.io/graphrag/config/yaml/
- LangGraph StateGraph: https://reference.langchain.com/python/langgraph/graph/state/StateGraph

## License

MIT
