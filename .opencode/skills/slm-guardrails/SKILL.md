---
name: slm-guardrails
description: Enforce SQLOutputContract Pydantic validation with sqlglot AST safety checks and the bounded 2-retry self-healing loop. Use before writing or modifying any SLM SQL generation, schema validation, or DuckDB execution code.
---

# SLM Guardrails

Project invariants from `AGENTS.md`. Load `pydantic-ai-agents` and `llm-guardrails` first for patterns, then apply these rules.

## Rules

1. Never pass raw Ollama strings to `duckdb.sql()`. Deserialize through `SQLOutputContract` (`src/backend/schemas/query.py`) using Pydantic v2 idioms (`model_validate`, `@field_validator`, `ConfigDict(extra="forbid", frozen=True)`).
2. Parse with `sqlglot.parse_one(sql, read="duckdb")`. Reject any root node that is not `sqlglot.exp.Select`.
3. Every `sqlglot.exp.Column` must exist in the target schema (`car_eval_clean.parquet` or `cali_housing_normalized.parquet`). Zero hallucinations.
4. On `ValidationError` or `duckdb.Error`, re-prompt at most 2 times (`max_retries = 2`) with the exact exception plus the valid column list.
5. After any change here, run `python3 scripts/validate_schemas.py`, `ruff`, and `mypy` immediately.
