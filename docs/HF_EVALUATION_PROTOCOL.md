# Hugging Face Multidimensional Evaluation Protocol (V2.5 research branch)

## Research claim and scope

GraphRAG originally targets **global query-focused summarization** (Edge et al.,
[arXiv:2404.16130](https://arxiv.org/abs/2404.16130)). Multi-hop QA is a
complementary diagnostic task, NOT an adequate reproduction of that paper's
global question preference results. Report each corpus/task separately.

**Implementation status:** Data adapters, versioned fixtures, paired mode
runner, deterministic answer/evidence metrics, graph-quality diagnostics,
robustness variants and paired bootstrap are implemented. **Real Hugging Face
datasets have not been downloaded or scored as part of CI.** CI uses synthetic
fixtures only. No quality/latency conclusions should be drawn from mock outputs.

## 1. Verified Hugging Face datasets

| Adapter | HF ID | Preferred split | Gold labels | Primary questions | Licensing |
|---|---|---|---|---|---|
| `hotpotqa` | `hotpotqa/hotpot_qa`, config `distractor` | validation | answer, supporting titles and sentence IDs | multi-hop bridge vs comparison; retrieval correctness | CC BY-SA 4.0 |
| `musique` | `awinml/musique` | validation | answer, `paragraphs.is_supporting`, question decomposition | 2/3/4-hop reasoning robustness | Verify original dataset conditions before use; card does not declare a license |
| `multi_news` | `alexfabbri/multi_news` | validation | source articles, human-written summary; **no per-article relevance labels** | global synthesis, coverage, redundancy | **Noncommercial research/education only; explicit `--accept-multi-news-terms`** |
| `qasper` | `allenai/qasper` | validation | answers plus textual evidence; evidence IDs not mapped to paragraph IDs in this adapter | paper-context QA / source referencing | CC BY 4.0 |

Dataset references:
- https://huggingface.co/datasets/hotpotqa/hotpot_qa
- https://huggingface.co/datasets/awinml/musique
- https://huggingface.co/datasets/alexfabbri/multi_news
- https://huggingface.co/datasets/allenai/qasper
- https://huggingface.co/datasets/galileo-ai/ragbench

RAGBench is recommended for a *separate* judge-calibration stage; the existing
`response`, `adherence_score` and support annotation describe an earlier
generated answer, **not a ground-truth score for graphRagX's newly generated
answer**. Do not transplant those labels.

Multi-News is read from the pinned repo's paired `data/val.src.cleaned`
and `data/val.tgt` files (or corresponding train/test paths), **not by
executing the legacy remote Python dataset script**. Pairing validates equal
line counts; `NEWLINE_CHAR` is converted back to real newlines. Review the
dataset usage agreement before download, and never commit source data to Git.

## 2. Full protocol: prepare > run > inspect > compare

Install:

```bash
pip install -r requirements.txt -r requirements-eval.txt
```

### A. Download immutable test fixtures

```bash
python -m scripts.hf_eval download \
  --dataset hotpotqa --split validation --count 20 --seed 42 \
  --output data/hf/hotpotqa-val-20.jsonl

python -m scripts.hf_eval download \
  --dataset musique --count 20 --seed 42 \
  --output data/hf/musique-val-20.jsonl

python -m scripts.hf_eval download \
  --dataset qasper --count 10 --seed 42 \
  --output data/hf/qasper-val-10.jsonl

# Only when research/educational use is permitted and terms reviewed:
python -m scripts.hf_eval download \
  --dataset multi_news --count 20 --accept-multi-news-terms \
  --output data/hf/multi-news-val-20.jsonl
```

The downloader resolves an immutable Hub revision SHA, records the selected
config, split, seed, source count, and file SHA-256 in a sidecar manifest. An
order-independent hash sampler avoids selecting just the first N rows.
`download` requires external connectivity and available dataset hosting; CI
does not perform it.

### B. Offline correctness smoke

```bash
python -m scripts.hf_eval run \
  --input data/hf/hotpotqa-val-20.jsonl \
  --conditions basic local global drift text_summary \
  --allow-mock-smoke \
  --output outputs/evaluation/hotpot-smoke.json
```

The result has `"mode": "offline_smoke"`. **Do not interpret** mock answers
as accuracy or semantic superiority. The compare command refuses smoke
reports.

### C. Real provider evaluation

Configure `LLM_PROVIDER=openai` and `EMBEDDING_PROVIDER=openai` with keys,
a suitable model, explicit token budgets, and resource controls. A compatible
local endpoint may also be configured with provider base URLs. Use the same
models and dimensions across all benchmark arms.

```bash
python -m scripts.hf_eval run \
  --input data/hf/hotpotqa-val-20.jsonl \
  --conditions basic local global drift text_summary C0 C1 C2 C3 TS SS \
  --top-k 5 \
  --output outputs/evaluation/hotpot-real.json
```

The six paper arms C0-C3/TS/SS are aliases for the appropriate Global
community level, text_summary, and basic mode. **Do not combine aliases**
with their equivalent modes in the same cost-normalized study; choose either
one labeled six-arm comparison or one practical five-mode study.

For paired bootstrap of two conditions on the *same cases*:

```bash
python -m scripts.hf_eval compare \
  --input outputs/evaluation/hotpot-real.json \
  --candidate global --baseline basic --dataset hotpotqa \
  --output outputs/evaluation/hotpot-paired.json
```

The benchmark attempts exactly one index per case and queries all conditions
against it; it rejects reuse when source hashes, dataset revision, embedding,
model, chunking or community algorithm differ. For a changed runtime, generate
separate index namespaces or explicitly request overwrite. Input caps are
enforced instead of truncating the corpus without disclosure.

### D. Perturbation evaluation

```bash
python -m scripts.hf_eval run \
  --input data/hf/hotpotqa-val-20.jsonl \
  --conditions basic local global drift \
  --variants original shuffle distractor drop_support \
  --variant-seed 42 --noise-documents 2 \
  --output outputs/evaluation/hotpot-robustness.json
```

- `shuffle`: permutes document order while preserving identity.
- `distractor`: inserts unrelated synthetic documents. Gold answers unchanged.
- `drop_support`: removes annotated supports, then **clears answer and gold
  retrieval labels**. This is a stress diagnostic, NOT an automatically
  unanswerable sample. Surviving paragraphs may still support an answer.
- Each perturbation gets its own content-addressed index ID; records store
  base case ID and perturbation seed.

### E. Generate an inspectable report

```bash
python -m scripts.hf_report \
  --input outputs/evaluation/hotpot-real.json \
  --output-dir outputs/evaluation/hotpot-report

# Optional charts; install matplotlib separately:
pip install matplotlib
python -m scripts.hf_report \
  --input outputs/evaluation/hotpot-real.json \
  --output-dir outputs/evaluation/hotpot-report --plots
```

This produces `summary.md` and `per_question.csv`, plus optional PNG
figures **separately for each dataset**. Offline smoke reports carry a
prominent warning and cannot be treated as accuracy measurements.

## 3. Seven dimensions, with interpretation constraints

| Axis | Implemented metrics | Supported task | Caveat |
|---|---|---|---|
| Answer | normalized Exact Match, token F1 | HotpotQA, MuSiQue, QASPER | Does not verify factual support; Chinese QA may need different normalization |
| Summarization | ROUGE-L F1 | Multi-News | Lexical overlap is not completeness, factuality or diversity |
| Evidence | document-level recall, precision, MRR, nDCG@k | HotpotQA, MuSiQue | Mapping global communities back to source docs measures *exposure*, not actual cited support |
| Search | community selection coverage, community cut size, included reports | Global, TS | High coverage may include distractors |
| Graph | entity/relation/claim counts, components, density, source-link coverage, complete hierarchy cut | All indexed cases | Structural proxies, not entity/relation ground-truth F1 |
| Engineering | index latency, query latency (mean/p95), failure rate, index reuse, DRIFT actions | All modes | Compare across matched resource budgets and hardware |
| Robustness | document shuffling, added distractors, support removal | Selected QA | Support-removal outcome is not a hallucination label |

Outputs contain question-level records and sliced aggregates by `condition`,
`dataset`, `task_type`, `difficulty`, `hop_count`, `question_type`,
and `robustness_variant`. Absent annotations are `null`, never a fake
perfect 1.0. **No measured monetary costs are emitted** without actual
usage instrumentation.

## 4. Statistical analysis

- Pair identical question IDs across candidate and baseline.
- Use per-question metric differences, not independent-sample unpaired tests.
- Resample question IDs with replacement (default 2,000 iterations, seed 42).
- Report paired effect, 95% percentile confidence interval, pair count,
  win/tie/loss and failure rate.
- Always stratify by dataset and task; do **not** average QA and summary
  metrics into one misleading benchmark leaderboard.
- For QASPER, multiple questions may share a paper; a paper-clustered
  bootstrap is recommended before making formal scientific significance
  claims. The current bootstrap is question-clustered.
- Multiple metric/model/ablation comparisons require error-rate correction
  (e.g., Holm), preregistered primary outcomes and independent test splits.

## 5. Suggested evaluation phases and acceptance gates

### E00: Static verification (zero model cost)
- Run all adapter, schema, metric, provenance and CLI safety tests.
- Check HotpotQA support titles, MuSiQue supporting paragraphs, QASPER answer
  annotation expansion and Multi-News equal line counts.
- Expected: all tests green, no live API access or dataset downloads.

### E01: Data-integrity pilot (20 cases/dataset)
- Download only validation snapshots; inspect source lengths, number of gold
  supports, outliers, answerability and per-dataset license.
- Expected: zero silent source omissions, immutable revision/checksum manifests.

### E02: Real-model pilot (20-50 cases/dataset)
- Same embedding/LLM, same dataset snapshot and reproducibility metadata.
- Log all failures and graph-quality metrics; rank by dataset + mode.
- Expected: no invalid source IDs, no falsely successful empty extraction,
  comparable model and index parameters.
- Treat costs/quality as provisional until token accounting is implemented.

### E03: Held-out multi-axis comparison (100-300 cases/dataset)
- Freeze corpora, model endpoints, prompt versions and experimental conditions.
- Report EM/F1, ROUGE-L, evidence metrics, latency, completeness,
  faithfulness/judged preference, confidence intervals and source audits.
- Reserve a separate human-audited sample for factuality and citation claims.
- Ensure enough evidence for formal conclusions; small pilots are exploratory.

### E04: GraphRAG-specific ablations
- `CHUNK_UNIT=characters` vs `tokens`; 300/600/1200 token windows,
  overlap 0/100/200 (valid combinations only).
- Louvain vs Leiden, resolution and hierarchy depth.
- Normalized exact entity resolution on/off; description merging and pruning.
- Claims on/off; vector vs graph retrieval; Global coverage vs relevance.
- Vary Global map budget, community level, DRIFT depth/actions, top-k and model.
- Never mix indexes generated under different ablations.

### E05: Production and safety evaluation
- API limits, index overwrite and concurrent requests.
- Multi-tenant access, permission-filtered retrieval, prompt-injection defenses.
- P50/P95/P99, throughput, indexing scalability, memory, embeddings/LLM
  token and monetary accounting, checkpoint recovery, corrupt-cache recovery.
- Build load tests separately from validity tests.

## 6. Priority gaps (not yet implemented)

1. **Measured** model token usage, actual pricing, per-stage traces, GPU
   memory/throughput and experiment-level spend controls.
2. Gold entity/relation annotations and claim-level citation verification.
3. Semantic summarization metrics beyond lexical ROUGE-L, verified with human
   inspection and calibrated judges.
4. Calibrated groundedness evaluators using RAGBench annotations (do not reuse
   original answers' labels on new outputs).
5. Paper-clustered bootstrap for QASPER, multiple-comparison correction,
   robustness paired-effect plots and independent repeat seeds.
6. A genuine **cross-case shared corpus index** and open-world retrieval
   benchmarks. Current HotpotQA/MuSiQue/QASPER evaluation uses **oracle
   per-question candidate contexts**, so its results are *not* end-to-end
   open-world retrieval scores.
7. A true paper Section 3.2 persona/task/global-question benchmark over a
   large corpus with audited global vs local question type.

## 7. Research data safety and repository hygiene

- Do not commit downloaded datasets, full-text paper/news corpora, keys or
  model responses containing sensitive material.
- Freeze sample selection, split, revision and source checksums.
- Respect CC BY-SA, CC BY, any unclear MuSiQue restrictions and the explicit
  Multi-News research-only agreement.
- Dataset samples may contain inaccurate or outdated facts; evaluation metrics
  are relative to dataset annotations, not independent fact-checking.
