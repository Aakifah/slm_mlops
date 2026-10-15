"""Build configuration invariant test.

The CI suite must stay free of internal documentation references, so this
module asserts only the public build configuration. Feature tests replace
and extend it as `src/` gains real code.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib


def test_pyproject_declares_supported_python() -> None:
    """The package metadata must parse and target the supported runtime."""
    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["project"]["name"]
    assert "3.11" in pyproject["project"]["requires-python"]
    assert sys.version_info >= (3, 11)
