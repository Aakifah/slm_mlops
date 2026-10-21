"""Min-max scaler fitting and deterministic serialization for the regression set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.backend.schemas.api import RegressionScalerSchema, ScalerBounds


def fit_scaler(
    train: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    dataset: str,
) -> RegressionScalerSchema:
    """Fit min/max bounds strictly on the training split.

    Args:
        train: The training split dataframe.
        numeric_columns: The numerical feature columns to bound.
        dataset: The target dataset name.

    Returns:
        A validated ``RegressionScalerSchema`` holding per-feature bounds.
    """
    features: dict[str, ScalerBounds] = {}
    for column in numeric_columns:
        values = pd.to_numeric(train[column], errors="coerce")
        features[column] = ScalerBounds(min=float(values.min()), max=float(values.max()))
    return RegressionScalerSchema(dataset=dataset, features=features)


def write_scaler(schema: RegressionScalerSchema, path: Path) -> None:
    """Serialize a scaler schema to disk.

    Args:
        schema: The validated scaler schema.
        path: The destination JSON path.
    """
    path.write_text(schema.model_dump_json(indent=2), encoding="utf-8")


def load_scaler(path: Path) -> RegressionScalerSchema:
    """Load and validate a persisted scaler artifact.

    Args:
        path: The source JSON path.

    Returns:
        The validated scaler schema.

    Raises:
        ValueError: If the on-disk artifact is malformed.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RegressionScalerSchema.model_validate(payload)
