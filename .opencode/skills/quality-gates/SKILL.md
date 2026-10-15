---
name: quality-gates
description: Run the blocking 5-stage local quality gate and threshold assertions before any commit or PR. Use when modifying src/, tests/, or scripts/, or before pushing.
---

# Quality Gates

Load `ml-testing`, `llm-evaluation`, `python-testing-patterns`, and `github-actions-templates` as needed.

## Gate

```bash
bash scripts/check.sh
```

Strict order: `ruff format --check` then `ruff check` then `mypy --config-file pyproject.toml` then `python3 scripts/validate_schemas.py` then `pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80`.

## Thresholds (`tests/quality_gates/test_eval_thresholds.py`)

- SQL execution success >= 85%
- Schema groundedness 100%
- Intent accuracy >= 92%
- Regression MAPE <= 10%

## Sequence

`scripts/check.sh` then `modal run scripts/modal_smoke_test.py` then `git push` then GitHub Actions then `docker compose up --build`.
