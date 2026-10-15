# SLM MLOps — Structured Data Extraction with Small Language Models

![quality_gates](https://github.com/Aakifah/slm_mlops/actions/workflows/quality_gates.yml/badge.svg)

End-to-end LLMOps system that answers natural-language questions over tabular
data. Simple point-prediction queries are routed to traditional ML baselines;
analytical queries go through a schema-aware RAG + Text-to-SQL pipeline backed
by small language models, with every generated query validated before execution.

## How it works

1. **Intent routing** — each question is classified as a single-record
   prediction (Random Forest / Gradient Boosting) or a complex analytical
   query (SLM Text-to-SQL over DuckDB).
2. **Guarded generation** — all model output is validated against strict
   response contracts: read-only `SELECT` statements only, zero unknown
   columns, with a bounded self-repair retry on failure.
3. **Semantic caching** — near-duplicate questions are served from a FAISS
   similarity cache without invoking a model.
4. **Evaluation gates** — SQL success rate, schema groundedness, routing
   accuracy, and regression error are asserted automatically on every change.

## Stack

FastAPI · Pydantic v2 · SQLGlot · FAISS · DuckDB · scikit-learn · Ollama
(`mistral:7b`, `qwen2.5:7b`, `llama3.2`) · React/Vite · Prometheus/Grafana ·
OpenTelemetry · Docker Compose

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
bash scripts/check.sh
```

`scripts/check.sh` runs the full local gate: ruff format, ruff lint,
mypy strict, schema validation, then pytest with coverage. The same gates run
in CI on every push and pull request.

## Layout

```text
src/backend/             FastAPI app (routes, middleware, schemas, services)
src/model_customization/ Intent router, semantic cache, RAG, SQL execution
src/data_preprocessing/  Dataset ingestion and cleaning pipelines
src/evaluator/           Baselines, benchmarks, and visualizations
tests/                   Unit tests and quality-gate threshold assertions
monitoring/              Prometheus, Alertmanager, and Grafana configs
scripts/                 Reference setup, schema validation, quality runner
```
