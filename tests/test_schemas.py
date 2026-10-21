"""Pydantic artifact contract tests for the data preparation schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.backend.schemas.api import (
    RAGChunk,
    RAGChunkSchema,
    RegressionScalerSchema,
    ScalerBounds,
)


def test_scaler_bounds_reject_degenerate() -> None:
    """Bounds must require a strictly increasing min/max interval."""
    with pytest.raises(ValidationError):
        ScalerBounds(min=1.0, max=1.0)
    with pytest.raises(ValidationError):
        ScalerBounds(min=2.0, max=1.0)


def test_scaler_bounds_accept_valid() -> None:
    """Valid bounds should round-trip without mutation."""
    bounds = ScalerBounds(min=0.0, max=100.0)
    assert bounds.min == 0.0
    assert bounds.max == 100.0


def test_scaler_bounds_forbid_extra() -> None:
    """Unknown fields must be rejected by the frozen contract."""
    with pytest.raises(ValidationError):
        ScalerBounds.model_validate({"min": 0.0, "max": 1.0, "scale": 2.0})


def test_regression_scaler_rejects_unknown_dataset() -> None:
    """The scaler must reference a supported dataset name."""
    with pytest.raises(ValidationError):
        RegressionScalerSchema(
            dataset="unknown",
            features={"price_usd": ScalerBounds(min=0.0, max=1.0)},
        )


def test_regression_scaler_json_roundtrip() -> None:
    """A scaler should serialize and deserialize losslessly."""
    schema = RegressionScalerSchema(
        dataset="sephora_pricing",
        features={"price_usd": ScalerBounds(min=3.0, max=395.0)},
    )
    restored = RegressionScalerSchema.model_validate_json(schema.model_dump_json())
    assert restored == schema


def test_rag_chunk_schema_rejects_unknown_dataset() -> None:
    """Chunks must reference a supported dataset name."""
    chunk = RAGChunk(id="x", dataset="unknown", text="t", columns=["a"], aliases=[])
    with pytest.raises(ValidationError):
        RAGChunkSchema(chunks=[chunk])


def test_build_rag_chunks_covers_both_datasets() -> None:
    """The built chunk set must cover both datasets with non-empty text."""
    from src.model_customization.rag.chunks import build_rag_chunks

    chunks = build_rag_chunks()
    assert len(chunks.chunks) > 0
    assert {chunk.dataset for chunk in chunks.chunks} == {
        "sephora_pricing",
        "skincare_compatibility",
    }
    assert all(chunk.text for chunk in chunks.chunks)
    assert all(chunk.columns for chunk in chunks.chunks)
