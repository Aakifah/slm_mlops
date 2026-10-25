"""Train-all-models helper function tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backend.services.ml_baseline import (
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
)
from src.data_preprocessing.preprocessing.parsers.synthetic import generate_synthetic
from src.evaluator.train_all_models import (
    _accuracy,
    _load,
    _mae,
    _mape,
    _prepare_classification,
    _prepare_regression,
    _rmse,
)


def test_accuracy() -> None:
    """Accuracy must be the fraction of exact label matches."""
    assert _accuracy(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0
    assert _accuracy(np.array([0, 1, 2]), np.array([0, 0, 2])) == pytest.approx(2 / 3)


def test_mape_rmse_mae() -> None:
    """Regression metrics must match their closed-form definitions."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 20.0, 27.0])
    assert _mape(y_true, y_pred) == pytest.approx((1 / 10 + 0 + 3 / 30) / 3 * 100)
    assert _rmse(y_true, y_pred) == pytest.approx(np.sqrt((1 + 0 + 9) / 3))
    assert _mae(y_true, y_pred) == pytest.approx((1 + 0 + 3) / 3)


def test_mape_handles_zero_target() -> None:
    """Zero targets must not divide by zero in MAPE."""
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 10.0])
    assert _mape(y_true, y_pred) == pytest.approx((1 / 1 + 0 / 10) / 2 * 100)


def test_prepare_classification() -> None:
    """Classification preparation must extract features and integer labels."""
    _, skincare = generate_synthetic(seed=1, n_pricing=20, n_skincare=20)
    features, labels = _prepare_classification(skincare)
    assert list(features.columns) == list(CLASSIFICATION_FEATURES)
    assert np.issubdtype(labels.dtype, np.integer)
    assert len(labels) == len(skincare)


def test_prepare_regression() -> None:
    """Regression preparation must extract features and float targets."""
    pricing, _ = generate_synthetic(seed=1, n_pricing=30, n_skincare=20)
    features, target = _prepare_regression(pricing)
    assert list(features.columns) == list(REGRESSION_FEATURES)
    assert np.issubdtype(target.dtype, np.floating)
    assert len(target) == len(pricing)


def test_load_reads_parquet(tmp_path: Path) -> None:
    """The loader must round-trip a Parquet file."""
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "sample.parquet"
    frame.to_parquet(path, index=False)
    loaded = _load(path)
    assert loaded.equals(frame)
