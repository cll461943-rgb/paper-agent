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
configs/                  Runtime configuration files
data/                     Small benchmark inputs and selected result artifacts
scripts/                  Utility scripts, including local FTS index building
src/scholar_agent/        Main package source
tests/                    Unit and evaluation-focused tests
evaluate.py               End-to-end recommendation evaluator
evaluate_pasa.py          Retrieval-only PaSa-style evaluator
model_details.md          Architecture and workflow notes
scholar_agent_v2.1_spec.md Project specification notes
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

Run retrieval-only PaSa-style evaluation:

```bash
python evaluate_pasa.py --dataset data/AutoScholarQuery_dev.jsonl --mode mock --limit 5
```

Useful live-evaluation options include provider selection, random sampling, per-case timeout, known-title clues, local FTS, and LLM disabling:

```bash
python evaluate_pasa.py --dataset data/AutoScholarQuery_dev.jsonl --mode live --providers pasa_local,openalex,arxiv,pubmed --sample-size 5 --sample-runs 3 --case-timeout-seconds 120 --enable-pasa-local-fts
```

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
