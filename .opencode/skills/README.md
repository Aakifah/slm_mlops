# Skills Index — slm_mlops

Project skills live in `.opencode/skills/<skill-id>/SKILL.md` (auto-discovered, project scope).
Routing rules come from `AGENTS.md` section "Skill Routing & Curated SKILL.md Catalog".

## Project-local (load these first for project invariants)

| Skill | When | Area |
|---|---|---|
| `slm-guardrails` | Pydantic schemas, sqlglot AST, self-healing retries, schema gates | `src/backend/schemas/`, `scripts/validate_schemas.py`, `sql_executor.py` |
| `intent-router` | Intent classifier, semantic cache, `references/` | `unified_router/intent_classifier.py`, `semantic_cache.py` |
| `quality-gates` | Any `src/`/`tests/`/`scripts/` change, pre-commit, pre-push | `scripts/check.sh`, `tests/quality_gates/`, `modal_smoke_test.py` |

## Research & browser (needs npm + Chromium)

| Skill | When | Backend |
|---|---|---|
| `web-research` | Current docs, APIs, fact-checking | `duckduckgo` + `fetch` MCP (connected) |
| `browser-automation` | JS-rendered pages, frontend verification | `playwright` MCP + Chromium (`npx -y playwright install chromium`) |
| `social-evidence-research` | Social-platform discussion with cited claims | upstream wshobson |

## Upstream (grouped by AGENTS.md routing matrix)

| Skill | Trigger | Upstream source |
|---|---|---|
| `fastapi` | REST endpoints, `Annotated` deps, auth, rate limiting, CORS | fastapi/fastapi |
| `pydantic-ai-agents` | Agents, tools, validators, streaming, multi-agent | Fuenfgeld/pydantic-ai-skills |
| `pydantic-evals` | Eval datasets, LLM-as-judge, model comparison | Fuenfgeld/pydantic-ai-skills |
| `llm-guardrails` | Input/output validation, PII, jailbreak prevention | timwukp/MLOps-agent-skills |
| `llm-rag` | FAISS index, chunking, hybrid search, reranking | timwukp/MLOps-agent-skills |
| `rag-implementation` | RAG systems with vector DBs (companion to llm-rag) | wshobson/llm-application-dev |
| `embedding-strategies` | Embedding model choice, chunking quality | wshobson/llm-application-dev |
| `llm-cost-optimization` | Semantic caching, routing, prompt compression | timwukp/MLOps-agent-skills |
| `llm-evaluation` | RAGAS, benchmarks, safety eval, A/B tests | timwukp/MLOps-agent-skills |
| `llm-observability` | Token/latency tracking, LangSmith, feedback | timwukp/MLOps-agent-skills |
| `llm-deployment` | Ollama, vLLM, quantization, serving infra | timwukp/MLOps-agent-skills |
| `llm-prompt-engineering` | Prompt patterns, versioning, injection defense | timwukp/MLOps-agent-skills |
| `ml-testing` | Quality gates, behavioral tests, CI integration | timwukp/MLOps-agent-skills |
| `ml-cicd` | GitHub Actions, promotion, rollback | timwukp/MLOps-agent-skills |
| `github-actions-templates` | Workflow templates for quality_gates.yml | wshobson/cicd-automation |
| `data-validation` | Great Expectations, Pandera, data contracts | timwukp/MLOps-agent-skills |
| `model-serving` | FastAPI/BentoML/Triton, Docker, A/B deploys | timwukp/MLOps-agent-skills |
| `model-monitoring` | Evidently, Prometheus metrics, alerting | timwukp/MLOps-agent-skills |
| `sql-optimization-patterns` | DuckDB SQL, indexing, EXPLAIN analysis | wshobson/developer-essentials |
| `error-handling-patterns` | Exceptions, Result types, graceful degradation | wshobson/developer-essentials |
| `code-review-excellence` | PR reviews, review standards (= code-reviewer) | wshobson/developer-essentials |
| `auth-implementation-patterns` | JWT/OAuth2, API key auth, RBAC | wshobson/developer-essentials |
| `debugging-strategies` | Root-cause analysis (= debugging-wizard) | wshobson/developer-essentials |
| `python-type-safety` | Type hints, generics, mypy strict | wshobson/python-development |
| `python-code-style` | Ruff, formatting, docstrings, no `#` comments | wshobson/python-development |
| `python-testing-patterns` | pytest, fixtures, mocking (= pytest skill) | wshobson/python-development |
| `python-error-handling` | Validation logic, exception hierarchies | wshobson/python-development |
| `distributed-tracing` | OTel spans across RAG/SLM/DuckDB (Cherry 4) | wshobson/observability-monitoring |
| `prometheus-configuration` | Metrics collection, alerting rules | wshobson/observability-monitoring |
| `grafana-dashboards` | Monitoring dashboards | wshobson/observability-monitoring |

## Notes

- AGENTS.md pointed `pytest` at `microsoft/debugpy`; that repo is a debugger, not a pytest skill, so `python-testing-patterns` + `ml-testing` cover pytest here.
- `python-pro` from the matrix is covered by the four `python-*` skills (type-safety for `mypy --strict`, code-style for `ruff` + no-`#`-comment rule).
- `code-reviewer` / `debugging-wizard` / `ablation-readme-rewriter` have no exact upstream IDs; mapped to `code-review-excellence` / `debugging-strategies` / `social-evidence-research` + project docs.
- Added beyond AGENTS.md for this stack: `llm-deployment` (Ollama), `model-serving`/`model-monitoring` (Docker/Prometheus), `distributed-tracing`/`prometheus-configuration`/`grafana-dashboards` (Cherry 4), `web-research`/`browser-automation` (npm + Chromium).
