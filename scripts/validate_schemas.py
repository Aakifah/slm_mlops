"""Validate Pydantic schema contracts and on-disk artifact schemas.

This standalone gate instantiates the artifact contracts against adversarial
payloads and the committed JSON artifacts. It runs before pytest in the quality
gate so malformed artifacts fail the build deterministically.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_ARTIFACTS_DIR: Path = _PROJECT_ROOT / "src" / "artifacts"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _expect_validation_error(action: Callable[[], object]) -> None:
    """Assert that an action raises a ``ValidationError``.

    Args:
        action: A callable that should raise ``ValidationError``.

    Raises:
        AssertionError: If the action does not raise.
    """
    try:
        action()
    except ValidationError:
        return
    raise AssertionError("invalid payload was accepted")


def _validate_contracts() -> None:
    """Exercise the artifact contracts against valid and adversarial payloads."""
    from src.backend.schemas.api import (
        RAGChunk,
        RAGChunkSchema,
        RegressionScalerSchema,
        ScalerBounds,
    )

    ScalerBounds.model_validate({"min": 0.0, "max": 1.0})
    _expect_validation_error(lambda: ScalerBounds.model_validate({"min": 1.0, "max": 1.0}))
    _expect_validation_error(lambda: ScalerBounds.model_validate({"min": 2.0, "max": 1.0}))

    RegressionScalerSchema.model_validate(
        {
            "dataset": "sephora_pricing",
            "features": {"price_usd": {"min": 3.0, "max": 395.0}},
        }
    )
    _expect_validation_error(
        lambda: RegressionScalerSchema.model_validate({"dataset": "unknown", "features": {}})
    )

    chunk = RAGChunk(
        id="sephora_pricing.price_usd",
        dataset="sephora_pricing",
        text="price_usd: Product price in US dollars.",
        columns=["price_usd"],
        aliases=["price", "cost"],
    )
    RAGChunkSchema.model_validate({"chunks": [chunk.model_dump()]})
    _expect_validation_error(
        lambda: RAGChunkSchema.model_validate(
            {
                "chunks": [
                    {"id": "x", "dataset": "unknown", "text": "t", "columns": [], "aliases": []}
                ]
            }
        )
    )


def _validate_artifacts() -> None:
    """Validate the committed JSON artifacts when they are present."""
    from src.backend.schemas.api import RAGChunkSchema, RegressionScalerSchema

    scaler_path = _ARTIFACTS_DIR / "regression_scaler.json"
    chunks_path = _ARTIFACTS_DIR / "rag_chunks.json"
    if scaler_path.exists():
        payload = json.loads(scaler_path.read_text(encoding="utf-8"))
        RegressionScalerSchema.model_validate(payload)
    if chunks_path.exists():
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        RAGChunkSchema.model_validate(payload)


def main() -> int:
    """Run the standalone contract and artifact validation gate.

    Returns:
        The process exit code.
    """
    _validate_contracts()
    _validate_artifacts()
    print("validate_schemas: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
