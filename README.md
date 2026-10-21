# Structured Data Extraction Using Small Language Models

![quality_gates](https://github.com/Aakifah/slm_mlops/actions/workflows/quality_gates.yml/badge.svg)

**Domain: Cosmetics & Skincare Intelligence** — *Derm-Safe Ingredient Triage &
Luxury Tax Price Predictor*.

An end-to-end LLMOps system that answers natural-language questions over tabular
beauty data. Simple point-prediction queries route to traditional ML baselines;
analytical queries go through a schema-aware RAG + Text-to-SQL pipeline backed
by small language models, with every generated query validated before execution.

## Milestone 1: Data Preparation & Management

Deterministic, leak-free, immutable artifacts for the Cosmetics & Skincare
domain, produced by a two-tier industrial ingestion strategy.

**Datasets**

| Dataset | Task | Target |
| :--- | :--- | :--- |
| `skincare_compatibility` | Classification | `compatibility_tier` (`Holy_Grail_Safe` / `Neutral` / `Irritation_Risk`) |
| `sephora_pricing` | Regression | `price_usd` |

Features are engineered from **real INCI ingredient lists** (detecting Retinol,
Salicylic Acid, Fragrance, Alcohol Denat, and comedogenic actives) and **real
product attributes** (size, rating, loves, exclusivity) to derive
`compatibility_tier` and `price_usd`.

**Two-tier ingestion**

1. **Primary** — pull the real Sephora catalogue (`MayaKitzis/sephora_products`
   `product_info.csv`, credential-free) or a locally staged CSV from
   `data/raw/`, then apply deterministic domain feature engineering.
2. **Fallback** — a seeded, domain-realistic synthesizer (real brands,
   log-normal price distributions, dermatological noise) that triggers only when
   running offline or when raw files/network are unavailable.

**Pipeline guarantees**

- Exact `(X, y)` deduplication and a stratified **70% / 15% / 15%**
  train/validation/test split with **zero cross-contamination** (product-level).
- Min-max scaler fitted **strictly on the training split** and serialized to
  `regression_scaler.json` for deterministic denormalization.
- Every JSON artifact is validated by a Pydantic v2 contract
  (`RegressionScalerSchema`, `RAGChunkSchema`).

**Generate the artifacts**

```bash
python -m src.data_preprocessing.preprocessing.main
python -m src.model_customization.rag.index_faiss
```

Set `SLM_MLOPS_OFFLINE=1` to force the deterministic synthetic fallback (used by
offline CI unit tests). All outputs are written read-only into `src/artifacts/`.

## How it works (target architecture)

1. **Intent routing** — classify each question as a single-record prediction or
   a complex analytical query.
2. **Guarded generation** — validate all model output against strict response
   contracts: read-only `SELECT` statements, zero unknown columns, bounded
   self-repair retry.
3. **Semantic caching** — serve near-duplicate questions from a FAISS similarity
   cache without invoking a model.
4. **Evaluation gates** — assert SQL success rate, schema groundedness, routing
   accuracy, and regression error on every change.

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

`scripts/check.sh` runs the full local gate: ruff format, ruff lint, mypy
strict, schema validation, then pytest with coverage. The same gates run in CI
on every push and pull request.

## Layout

```text
src/backend/             FastAPI app (routes, middleware, schemas, services)
src/model_customization/ Intent router, semantic cache, RAG, SQL execution
src/data_preprocessing/  Dataset ingestion, feature engineering, splitting
src/evaluator/           Baselines, benchmarks, and visualizations
src/artifacts/           Read-only Parquet splits, scaler, FAISS index, models
tests/                   Unit tests and quality-gate threshold assertions
monitoring/              Prometheus, Alertmanager, and Grafana configs
scripts/                 Reference setup, schema validation, quality runner
```
