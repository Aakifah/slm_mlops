"""Repository skeleton invariant tests.

These tests guard the project scaffolding itself: required files exist,
configuration parses, quality gates stay in sync between local and CI
runners, and reference checkouts remain git-metadata free. They are
replaced incrementally by feature tests as `src/` gains real code.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    ".env",
    ".env.example",
    "AGENTS.md",
    "opencode.json",
    "scripts/check.sh",
    "scripts/validate_schemas.py",
    "scripts/pull_references.sh",
    ".github/workflows/quality_gates.yml",
    "references/README.md",
    "references/DECISION.md",
]

EXPECTED_MCP_SERVERS = {"duckduckgo", "fetch", "arxiv", "playwright", "huggingface"}

GATE_COMMANDS = [
    "ruff format",
    "ruff check",
    "mypy --config-file pyproject.toml",
    "scripts/validate_schemas.py",
    "pytest",
]


def test_python_runtime_is_supported() -> None:
    """The test runner itself must satisfy the floor declared in pyproject."""
    assert sys.version_info >= (3, 11)


def test_required_skeleton_files_exist() -> None:
    """Every file the scaffolding promises must be present."""
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    assert not missing, f"missing skeleton files: {missing}"


def test_env_file_is_empty() -> None:
    """The committed `.env` must stay empty; real values live outside git."""
    assert (ROOT / ".env").stat().st_size == 0


def test_opencode_config_parses_with_expected_servers() -> None:
    """The MCP configuration must declare the full server set."""
    config = json.loads((ROOT / "opencode.json").read_text(encoding="utf-8"))
    servers = config["mcp"]["servers"]
    assert EXPECTED_MCP_SERVERS <= set(servers)


def test_pytest_config_enforces_coverage_gate() -> None:
    """The coverage gate threshold must remain configured at eighty percent."""
    with open(ROOT / "pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--cov-fail-under=80" in addopts


def test_local_quality_runner_covers_all_gates() -> None:
    """`scripts/check.sh` must invoke every blocking gate in order."""
    content = (ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")
    positions = [content.index(command) for command in GATE_COMMANDS]
    assert positions == sorted(positions)


def test_ci_workflow_covers_all_gates() -> None:
    """The GitHub workflow must enforce the same gates as the local runner."""
    workflow = (ROOT / ".github" / "workflows" / "quality_gates.yml").read_text(encoding="utf-8")
    for command in GATE_COMMANDS:
        assert command in workflow


def test_reference_checkouts_have_no_git_metadata() -> None:
    """Cloned references must not carry `.git` dirs into source control."""
    references = ROOT / "references"
    nested = [path for path in references.iterdir() if path.is_dir() and (path / ".git").exists()]
    assert not nested, f"reference dirs contain .git: {nested}"


def test_primary_reference_checkouts_exist() -> None:
    """The research verdict requires semantic-router and setfit checkouts."""
    for name in ("semantic-router", "setfit"):
        assert (ROOT / "references" / name).is_dir(), f"missing references/{name}"
