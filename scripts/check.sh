#!/usr/bin/env bash
set -eo pipefail
ruff format src tests scripts --check
ruff check src tests scripts
mypy --config-file pyproject.toml src tests scripts
python scripts/validate_schemas.py
PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80
