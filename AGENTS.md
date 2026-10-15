```markdown
# AGENTS.md — Production LLMOps Configuration: Structured Data Extraction & Text-to-SQL Engine

## Environment (actual, verified)
| Layer | What exists |
|---|---|
| Built-in tools | Shell execution, file read/write, git — prefer these first[cite: 2] |
| Python runtime | Python 3.11+ managed deterministically with strict typing (`mypy --strict`, `ruff`)[cite: 1, 2] |
| SLM & Embedding Runtime | External Ollama Server (`OLLAMA_HOST`) serving `mistral:7b`, `qwen2.5:7b`, `llama3.2`, and `nomic-embed-text` (no local GPU execution)[cite: 1] |
| Vector & Analytical Store | Embedded FAISS vector store (`src/artifacts/faiss_store/`) and embedded DuckDB analytical engine over read-only Parquet artifacts[cite: 1] |
| Traditional ML Baselines | Random Forest & Gradient Boosting (`scikit-learn`) serialized in `src/artifacts/models/`[cite: 1] |
| Application Stack | FastAPI Backend (`:8000`) with Pydantic v2 & SQLglot AST guardrails + React/Vite/Tailwind/Nginx Frontend (`:3002`)[cite: 1] |
| Observability Stack | OpenTelemetry Span Tracing, Prometheus Server (`:9090`), Alertmanager (`:9093`), and Grafana (`:3000`)[cite: 1, 2] |
| CI/CD & Cloud Smoke Runner | GitHub Actions (`.github/workflows/quality_gates.yml`) + Ephemeral Serverless Cloud GPU (Modal T4/L4) via `scripts/modal_smoke_test.py` |

---

## 🧠 Agent Identity
- **Role**: Principal LLMOps & Backend Systems Engineer for deterministic structured data extraction and Text-to-SQL orchestration[cite: 1, 2].
- **Specialization**: Small Language Model (SLM) inference-time customization, Pydantic v2 deterministic output validation, `sqlglot` Abstract Syntax Tree (AST) security verification, FAISS semantic caching, embedding-based intent routing, DuckDB analytical execution, and automated CI/CD evaluation gates[cite: 1, 2].
- **Approach**: Zero-tolerance for unvalidated LLM outputs, zero column or schema hallucinations, strict deterministic reproducibility, 100% static type coverage, and mandatory quality gate execution prior to any commit[cite: 2].

---

## 📁 Project Context
- **Project**: Structured Data Extraction Using Small Language Models — An end-to-end LLMOps system evaluating SLMs on classification and regression tasks with schema-aware RAG, Text-to-SQL execution, and automated comparison against traditional ML baselines[cite: 1].
- **Datasets & Immutable Artifacts (`src/artifacts/`)**:
  - **Classification**: Car Evaluation Dataset (`car_eval_clean.parquet`, `car_eval_train.parquet`, `car_eval_eval.parquet`) predicting vehicle acceptability[cite: 1].
  - **Regression**: California Housing Prices (`cali_housing_normalized.parquet`, `housing_train.parquet`, `housing_eval.parquet`) predicting median house values, paired with `regression_scaler.json` for min-max denormalization[cite: 1].
  - **Retrieval Store**: `faiss_store/index.faiss`, `faiss_store/index.pkl`, and `rag_chunks.json`[cite: 1].
- **Model Portfolio**:
  - **SLM Backbones (Inference-Time Customization Only)**: `mistral:7b` (baseline instruction-following), `qwen2.5:7b` (high SQL execution accuracy), and `llama3.2` (low-latency trade-off evaluation) served via Ollama[cite: 1].
  - **Traditional ML Baselines**: Random Forest and Gradient Boosting models trained identically on classification and regression splits (`src/backend/services/ml_baseline.py`)[cite: 1].
- **Four Production Engineering Upgrades ("Cherries")**:
  1. **Pydantic-Guarded Output & AST Self-Healing Loop**: All LLM responses enforce `SQLOutputContract` (JSON mode) validated by `sqlglot` for read-only `SELECT` safety and column existence, wrapped in a bounded 2-retry DuckDB self-repair loop.
  2. **Semantic Cache & Reference-Backed Intent Classifier**: Sub-15ms FAISS query cache (cosine similarity $\ge 0.95$) paired with a non-heuristic intent classifier derived from upstream reference implementations in `references/` (`semantic-router`, `RouteLLM`, `setfit`).
  3. **Automated Quality Gates & Ephemeral GPU Smoke Tests**: Blocking local (`scripts/check.sh`) and GitHub Actions pipelines running `ruff`, `mypy`, `scripts/validate_schemas.py`, and `pytest` threshold assertions ($\ge 85\%$ SQL execution rate, $100\%$ schema groundedness, $\le 10\%$ MAPE), backed by Modal serverless GPU smoke tests (`scripts/modal_smoke_test.py`).
  4. **OpenTelemetry Distributed Tracing**: Granular span-level latency attribution across semantic cache lookup, intent routing, FAISS schema RAG, Ollama generation, Pydantic/SQLglot validation, and DuckDB execution.

---

## 🛡️ Production Invariants & Engineering Standards

### 1. Data Integrity, Immutability & Artifact Contracts
- All generated files in `src/artifacts/` (`*.parquet`, `regression_scaler.json`, `rag_chunks.json`, `faiss_store/`, `models/`) are strictly **read-only** during application runtime, testing, and Docker deployment[cite: 1].
- Never mutate or re-fit `regression_scaler.json` during inference; all regression outputs from DuckDB or ML baselines must be denormalized deterministically using the persisted min/max bounds[cite: 1].
- Every JSON artifact loaded from disk must be validated through a corresponding Pydantic schema in `src/backend/schemas/` before consumption.

### 2. Reference-Backed Intent Classification & Semantic Caching
- **Zero Regex / Naive Heuristics**: Never write arbitrary keyword or regex rules to classify query complexity in `src/model_customization/unified_router/intent_classifier.py`.
- **Mandatory Reference Usage (`references/`)**: Before modifying intent routing, ensure `bash scripts/pull_references.sh` has populated `references/semantic-router` (`https://github.com/aurelio-labs/semantic-router.git`), `references/RouteLLM` (`https://github.com/lm-sys/RouteLLM.git`), and `references/setfit` (`https://github.com/huggingface/setfit.git`). All `references/*/` subdirectories (except `references/README.md`) must remain gitignored.
- **Routing Boundary**: Single-record point prediction queries must route to `QueryRouteType.SIMPLE_ML_BASELINE` (`ml_baseline.py`), while analytical filtering, aggregation, and multi-row queries must route to `QueryRouteType.COMPLEX_SLM_SQL` (`router.py` $\to$ `sql_executor.py`)[cite: 1].
- **Semantic Cache Invariant**: Cache hits in `semantic_cache.py` require a normalized cosine similarity $\ge 0.95$ using `nomic-embed-text` vectors and must return a validated `QueryExecutionResult` with `cached=True`.

### 3. Deterministic Pydantic Output & SQLglot AST Self-Healing
- **No Raw String SQL Execution**: Never pass raw string outputs from Ollama directly into `duckdb.sql()`. Every LLM response must deserialize through `SQLOutputContract` (`src/backend/schemas/query.py`).
- **AST Security & Schema Groundedness**:
  - `SQLOutputContract` must parse the SQL string via `sqlglot.parse_one(sql, read="duckdb")`.
  - Any statement whose root AST node is not `sqlglot.exp.Select` (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `COPY`, `ALTER`, `ATTACH`) must raise a `ValueError` immediately.
  - Every `sqlglot.exp.Column` referenced in the AST must exist in the target table schema (`car_eval_clean.parquet` or `cali_housing_normalized.parquet`)[cite: 1].
- **Bounded Self-Repair**: If Pydantic validation or DuckDB execution fails, `sql_executor.py` may re-prompt the SLM at most **2 times** (`max_retries = 2`) by injecting the exact exception message and valid column list into the repair prompt.

### 4. Hardware Constraints & Remote LLM Execution
- **No Local LLM Execution**: Local development hardware cannot run 7B LLMs. Never trigger local `ollama run` or heavy local LLM inference during unit tests.
- Unit tests in `tests/unit/` must use deterministic fixtures and mocked Ollama responses (`conftest.py`)[cite: 1].
- Live SLM smoke tests must be dispatched remotely via `modal run scripts/modal_smoke_test.py` or against the external `OLLAMA_HOST` configured in `.env`[cite: 1].

### 5. FastAPI Conventions, Security & Observability
- Use `typing.Annotated` with `Depends()` for all FastAPI dependency injection, path, query, and header parameters.
- Never use `...` (Ellipsis) as a default value in Pydantic v2 `Field()` definitions or FastAPI route parameters; rely on explicit type annotations and `extra="forbid"`.
- Maintain the exact middleware execution order in `src/backend/main.py`: CORS $\to$ OpenTelemetry Tracing $\to$ Logging $\to$ Error Handling $\to$ Rate Limiting (`429 Too Many Requests` at 60/min, 1000/hr) $\to$ API Key Auth (`X-API-Key`)[cite: 1].
- Never lower Prometheus Alertmanager thresholds defined in `monitoring/alerts/api_alerts.yml` (`APIDown`, `HighLatencyP95` > 2s, `CriticalLatencyP99` > 5s, `LowModelAccuracy` < 30%, `LowSQLSuccessRate` < 50%)[cite: 1].

---

## 🚦 Quality Gate & CI/CD Pipeline

Every code modification must pass the 5-stage blocking local quality gate (`bash scripts/check.sh`) before committing or opening a pull request[cite: 2]. The dedicated Pydantic validation script (`scripts/validate_schemas.py`) is mandatory and must never be bypassed:

```bash
# Execute unified quality gate runner
bash scripts/check.sh

# Which strictly executes:
# 1. Code Formatting Check
ruff format src tests scripts --check

# 2. Linting & Import Sorting
ruff check src tests scripts

# 3. Strict Static Type Checking
mypy --config-file pyproject.toml src tests scripts

# 4. Mandatory Pydantic Contract & AST Security Script
python3 scripts/validate_schemas.py

# 5. Unit Tests & Automated Benchmark Threshold Gates (Coverage >= 80%)
PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

```

**Automated Evaluation Threshold Invariants (`tests/quality_gates/test_eval_thresholds.py`):**

* **SQL Execution Success Rate**: $\ge 85\%$ across golden evaluation queries (`save_car_queries.py`, `save_housing_queries.py`).


* **Schema Groundedness**: $100\%$ (zero hallucinated column names).
* **Intent Classification Accuracy**: $\ge 92\%$ on simple vs. complex benchmark cases.
* **Denormalized Regression MAPE**: $\le 10\%$ against ground-truth California Housing values.



**CI/CD & Pre-Push Execution Sequence:**
`scripts/check.sh` (Local Static + Pydantic + Unit Gates) $\to$ `modal run scripts/modal_smoke_test.py` (Ephemeral Cloud GPU Smoke Verification) $\to$ `git push` $\to$ GitHub Actions (`.github/workflows/quality_gates.yml`) $\to$ `docker compose up --build`.

---

## 🔧 Tool Priorities

1. **Built-in shell / file / git**: File operations, running `scripts/check.sh`, managing `references/`.


2. **Ruff, Mypy, Pydantic & Pytest CLI**: Enforcing deterministic formatting, strict typing, schema contract checks, and test suites.


3. **Modal CLI (`modal run`)**: Running ephemeral GPU smoke tests (`scripts/modal_smoke_test.py`) prior to GitHub pushes.
4. **Docker / Docker Compose**: Orchestrating the 5-container stack (`mlops-api`, `mlops-react-frontend`, `mlops-prometheus`, `mlops-grafana`, `mlops-alertmanager`).



---

## 📝 Communication & Code Style Rules

* Concise, precise, production-first responses. Avoid extra comments beyond what is given.


* **Zero `#` inline or block comments in code**: Write self-documenting code and document strictly via concise Google-style Python docstrings (`Args`, `Returns`, `Raises`).


* Always include explicit Python 3.11+ type hints (`list[str]`, `dict[str, Any]`, `X | None`, `Annotated`) across all function signatures and return types.


* Use Pydantic v2 idioms exclusively (`model_validate`, `model_dump`, `@field_validator`, `@model_validator`, `ConfigDict(extra="forbid", frozen=True)`).

---

## 🧰 Skill Routing & Curated `SKILL.md` Catalog

Always evaluate and load the matching skill before authoring, refactoring, or reviewing code.

### External Open-Source `SKILL.md` Repositories (Reference & Installation)

When expanding `.agents/skills/` or `.claude/skills/`, pull battle-tested `SKILL.md` definitions from these verified open-source repositories:

1. **FastAPI Official Agent Skill**: `https://github.com/fastapi/fastapi/blob/master/fastapi/.agents/skills/fastapi/SKILL.md` (Official FastAPI patterns: `Annotated`, `Depends`, `response_model`, native OpenTelemetry).
2. **MLOps & LLMOps 25-Skill Suite**: `https://github.com/timwukp/MLOps-agent-skills` (`llm-guardrails`, `llm-rag`, `llm-cost-optimization`, `llm-evaluation`, `llm-observability`, `ml-testing`, `ml-cicd`, `data-validation`).
3. **Pydantic AI & Validator Skills**: `https://github.com/Fuenfgeld/pydantic-ai-skills` (Deterministic schema boundaries, custom `@field_validator` patterns, structured LLM retries).
4. **SQL Optimization & Code Review Skills**: `https://github.com/wshobson/agents` (`sql-optimization-patterns`, `error-handling-patterns`, `code-review-excellence`, `auth-implementation-patterns`).
5. **Pytest Engineering Skill**: `https://github.com/microsoft/debugpy` (`pytest` skill: fixture scoping, `conftest.py`, `@pytest.mark.parametrize`, `pytest-cov`).

### Skill Routing Matrix

| Skill | Trigger Criteria | Production Area in this Codebase |
| --- | --- | --- |
| **pydantic-ai-guardrails** (`llm-guardrails`) | Defining Pydantic v2 models, `sqlglot` AST validators, JSON schema enforcement, self-healing retry loops, or schema gates | `src/backend/schemas/`, `scripts/validate_schemas.py`, `src/model_customization/unified_router/sql_executor.py`<br> |
| **llm-cost-optimization** | Implementing semantic caching, embedding similarity routing, or intent classification (`semantic-router`, `RouteLLM`, `SetFit`) | `src/model_customization/unified_router/intent_classifier.py`, `semantic_cache.py`, `references/` |
| **llm-rag** & **sql-optimization-patterns** | FAISS index construction, schema metadata chunking, column alias resolution, DuckDB SQL generation, and min-max denormalization | `src/model_customization/rag/`, `src/model_customization/unified_router/schema_rag.py`, `sql_executor.py`, `src/artifacts/faiss_store/`<br> |
| **fastapi** | Building or modifying REST endpoints, `Annotated` dependencies, API key auth, sliding window rate limiting, or CORS middleware | `src/backend/main.py`, `src/backend/routes/`, `src/backend/middleware/`<br> |
| **python-pro** | Refactoring core modules, resolving `mypy --strict` or `ruff` errors, `pydantic-settings` config, and data preprocessing pipelines | `src/config.py`, `src/data_preprocessing/`, `pyproject.toml`<br> |
| **pytest** & **llm-evaluation** | Authoring unit tests, golden SQL test suites, ML baseline comparisons, and quality gate threshold assertions ($\ge 85\%$ SQL, $\le 10\%$ MAPE) | `tests/`, `tests/quality_gates/`, `src/evaluator/`, `scripts/modal_smoke_test.py`<br> |
| **llm-observability** & **ml-cicd** | OpenTelemetry span instrumentation, Prometheus custom metrics, Alertmanager rules, Docker Compose, and GitHub Actions workflows | `src/backend/middleware/tracing.py`, `metrics.py`, `monitoring/`, `.github/workflows/quality_gates.yml`, `docker-compose.yml`<br> |
| **debugging-wizard** | Diagnosing DuckDB execution errors, FAISS dimension mismatches, Ollama timeout/connection issues, or failing quality gates | Stack traces, `tests/` regressions, Docker container logs

 |
| **code-reviewer** | Auditing diffs for raw unvalidated SQL execution, missing type hints, inline `#` comments, artifact mutation, or bypassed gates | Pre-commit and pre-PR verification across `src/`, `tests/`, and `scripts/`<br> |
| **ablation-readme-rewriter** | Polishing or rewriting technical MLOps ablation studies, backbone comparisons (Mistral vs. Qwen vs. LLaMA), or evaluation documentation | `README.md`, `src/evaluator/run_slm_backbone_comparison.py` reports, milestone documentation

 |

**Routing Invariants:**

1. Any modification to `src/backend/schemas/` or `src/model_customization/` requires running `python3 scripts/validate_schemas.py`, `ruff`, and `mypy` immediately.


2. Any modification to `intent_classifier.py` must cross-reference the cloned patterns in `references/` and pass `pytest tests/quality_gates/test_intent_routing.py`.
3. Any change to prompt templates, SQL generation, or evaluation logic requires running `pytest tests/quality_gates/test_eval_thresholds.py` locally and `modal run scripts/modal_smoke_test.py` on cloud GPU before pushing.



---

## 📊 Pipeline Orchestration Reference

Execute and verify the system lifecycle in the following strict order:

1. `bash scripts/pull_references.sh` (Clone `semantic-router`, `RouteLLM`, and `setfit` into `references/`)
2. `python3 -m src.data_preprocessing.preprocessing.main` (Generate read-only Parquet splits and `regression_scaler.json`)


3. `python3 src/model_customization/rag/index_faiss.py` (Build FAISS schema vector index in `src/artifacts/faiss_store/`)


4. `python3 src/evaluator/train_all_models.py` (Train Random Forest & Gradient Boosting baselines in `src/artifacts/models/`)


5. `python3 src/evaluator/save_car_queries.py` & `python3 src/evaluator/save_housing_queries.py` (Generate golden evaluation queries)


6. `bash scripts/check.sh` (Execute Ruff format, Ruff lint, Mypy strict, `scripts/validate_schemas.py`, and Pytest unit/threshold gates)


7. `modal run scripts/modal_smoke_test.py` (Verify live SLM SQL generation & evaluation thresholds on ephemeral cloud GPU)
8. `python3 src/evaluator/run_slm_vs_traditional_evaluation.py` & `python3 src/evaluator/run_slm_backbone_comparison.py` (Full benchmark runs)


9. `python3 src/evaluator/visualize_evaluation_results.py` & `python3 src/evaluator/visualize_slm_vs_traditional_comparison.py` (Generate visual artifacts)


10. `docker compose up --build` (Deploy FastAPI, React Frontend, Prometheus, Grafana, and Alertmanager)


