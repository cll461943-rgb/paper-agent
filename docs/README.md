# Scholar Agent

Scholar Agent is a Python research-paper search and recommendation agent. It turns a natural-language academic query into a structured intent contract, retrieves candidate papers from local and online scholarly sources, reranks them with constraint-aware evidence checks, and emits a structured recommendation report.

The project is currently focused on benchmark-driven improvements for scholarly retrieval and recommendation, especially PaSa/AutoScholar-style recall evaluation, budget control, multi-source retrieval, and precision-preserving final synthesis.

## What It Does

- Parses complex research questions into structured `QueryPlan` constraints.
- Generates and reviews search queries across multiple retrieval rounds.
- Retrieves candidates from mock data, local PaSa-style paper indexes, OpenAlex, arXiv, PubMed, and Semantic Scholar.
- Expands candidates through reference/citation chains when configured.
- Deduplicates papers by title, DOI, arXiv id, and provider identifiers.
- Selects evidence with LLM assistance, then validates evidence locally to reduce hallucinated support.
- Reranks papers with relevance, constraint coverage, evidence completeness, source agreement, recency, authority, and graph-prior signals.
- Runs benchmark scripts for final recommendation metrics and retrieval-only PaSa recall.

## Repository Layout

```text
configs/                  Runtime configuration files (default, effect_first, high_recall, ablations)
data/                     Benchmark datasets, cache, local paper index, and dev/test sets
  benchmarks/             AutoScholarQuery, RealScholarQuery, id2paper.json, litsearch
  cache/                  API response cache (papers/, queries/) — do not delete
  cs_paper_2nd/           Local PaSa paper corpus (JSON keyed by normalized title)
  开发+测试集/             CNScholarQuery ZH v0.4 Chinese query dev/test sets
docs/                     Project development guide and design docs
outputs/                  Benchmark result artifacts
scripts/                  Utility scripts (build_pasa_local_fts.py)
src/scholar_agent/        Main package source
tests/                    Unit and evaluation-focused tests
前期调研/                  Competition rules and background materials
evaluate.py               End-to-end recommendation evaluator (supports -f for custom datasets)
evaluate_pasa.py          Retrieval-only PaSa-style evaluator
eval_pool_recall_temp.py  Pool recall benchmark script
eval_ablation.py          Ablation experiment runner
run_ablation.py           Ablation orchestration script
```

Large generated artifacts are intentionally not versioned: `data/cache/`, `*.sqlite`, run logs, PID files, Python caches, and local `.env` files.

## Installation

Use Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Configuration

Copy the example environment file and fill in only the keys you need:

```bash
cp .env.example .env
```

The default configuration runs in mock mode:

```bash
python -m scholar_agent.cli "papers about retrieval augmented generation" --mode mock
```

For live retrieval and LLM-assisted evidence selection, configure `DEEPSEEK_API_KEY` or another OpenAI-compatible endpoint in `.env`, then use a live config:

```bash
python -m scholar_agent.cli "recent papers on graph neural networks for traffic prediction" --mode live --config configs/default.yaml --output outputs/report.json
```

## Local PaSa Index

`pasa_local` expects local data under `data/`:

- `id2paper.json` maps paper ids to titles.
- `cs_paper_2nd.zip` contains paper JSON payloads keyed by normalized title.
- `pasa_local_fts.sqlite` is an optional generated FTS sidecar for abstract/reference search.

The SQLite sidecar is large and ignored by Git. Rebuild it locally with:

```bash
python scripts/build_pasa_local_fts.py --root data
```

## Evaluation

Run the test suite:

```bash
pytest
```

Run the end-to-end evaluator:

```bash
python evaluate.py --mode mock --limit 5 --output data/evaluation_sample.json
```

Specify a custom dataset with `-f` (e.g. CNScholarQuery Chinese query sets):

```bash
# CNScholarQuery real50 (avg_gold=15.8, use high_recall config to avoid K truncation)
python evaluate.py -f data/开发+测试集/CNScholarQuery_ZH_real50_polished.jsonl --config configs/high_recall.yaml -c 1

# CNScholarQuery test_250
python evaluate.py -f data/开发+测试集/CNScholarQuery_ZH_test_250_polished.jsonl --config configs/effect_first.yaml -c 1-5

# CNScholarQuery dev_1000 (large-scale Chinese query development set)
python evaluate.py -f data/开发+测试集/CNScholarQuery_ZH_v0.4_dev_1000.jsonl --config configs/default.yaml --limit 10
```

Run retrieval-only PaSa-style evaluation:

```bash
python evaluate_pasa.py --dataset data/AutoScholarQuery_dev.jsonl --mode mock --limit 5
```

Useful live-evaluation options include provider selection, random sampling, per-case timeout, known-title clues, local FTS, and LLM disabling:

```bash
python evaluate_pasa.py --dataset data/AutoScholarQuery_dev.jsonl --mode live --providers pasa_local,openalex,arxiv,pubmed --sample-size 5 --sample-runs 3 --case-timeout-seconds 120 --enable-pasa-local-fts
```

### Configuration Presets

| Config | k_max | hard_max_output | Use case |
|--------|-------|-----------------|----------|
| `default.yaml` | 20 | 12 | General development, AutoScholarQuery (avg_gold 2-3) |
| `effect_first.yaml` | 20 | 12 | LLM-led effect-first pipeline, Pro model |
| `high_recall.yaml` | 40 | 40 | CNScholarQuery real50 (avg_gold=15.8, max=65) and other high-gold datasets |

The `k_max` controls the F1-aware controller's search ceiling; `hard_max_output` caps total output including recall guard. Raise both for datasets with many gold papers to avoid top-K truncation.

## Main Components

- `planning`: query understanding, query generation, result review, and strategy optimization.
- `retrieval`: provider abstraction and integrations for mock, local PaSa, OpenAlex, arXiv, PubMed, Semantic Scholar, and reference expansion.
- `selection`: LLM evidence selection plus local evidence validation.
- `ranking`: dynamic-k and final reranking.
- `synthesis`: final report synthesis and recommendation capping.
- `evaluation`: metrics, error analysis, top-k sweeps, and PaSa scoring helpers.
- `workflow`: budget tracking and the end-to-end `PaperAgentPipeline`.

## Safety Notes

- Do not commit `.env` or real API keys.
- Keep generated caches and SQLite indexes local.
- Prefer benchmark evidence before claiming retrieval-quality improvements.
- Use mock mode for quick smoke tests and live mode only when credentials and network access are ready.
