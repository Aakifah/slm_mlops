"""Train and evaluate the five traditional ML baselines for both datasets.

Loads the read-only Parquet splits, fits each model on the 70% training split,
reports metrics on the 15% validation and holdout test splits, and persists the
fitted pipelines into ``src/artifacts/models/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.backend.services.ml_baseline import (
    CLASS_TO_INDEX,
    CLASSIFICATION_FEATURES,
    MODEL_NAMES,
    REGRESSION_FEATURES,
    build_classification_models,
    build_regression_models,
    model_filename,
)
from src.data_preprocessing.preprocessing.paths import (
    ARTIFACTS_DIR,
    DETERMINISTIC_SEED,
    EVALUATION_RESULTS_DIR,
    MODELS_DIR,
)
from src.data_preprocessing.preprocessing.schemas import PRICING_TARGET, SKINCARE_TARGET


def _load(path: Path) -> pd.DataFrame:
    """Read a Parquet split into a dataframe.

    Args:
        path: The Parquet file path.

    Returns:
        The loaded dataframe.
    """
    return pd.read_parquet(path)


def _accuracy(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground-truth encoded labels.
        y_pred: Predicted encoded labels.

    Returns:
        The fraction of correct predictions.
    """
    return float(np.mean(np.ravel(y_true) == np.ravel(y_pred)))


def _mape(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
    """Compute mean absolute percentage error in percent.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        The MAPE as a percentage.
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    denominator = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def _rmse(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
    """Compute root mean squared error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        The RMSE.
    """
    return float(np.sqrt(np.mean((np.ravel(y_true) - np.ravel(y_pred)) ** 2)))


def _mae(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
    """Compute mean absolute error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        The MAE.
    """
    return float(np.mean(np.abs(np.ravel(y_true) - np.ravel(y_pred))))


def _prepare_classification(frame: pd.DataFrame) -> tuple[pd.DataFrame, NDArray[np.int_]]:
    """Extract classification features and encoded labels.

    Args:
        frame: A skincare compatibility dataframe.

    Returns:
        A tuple of the feature dataframe and integer-encoded labels.
    """
    features = frame[list(CLASSIFICATION_FEATURES)]
    labels = frame[SKINCARE_TARGET].map(CLASS_TO_INDEX).to_numpy(dtype=int)
    return features, labels


def _prepare_regression(frame: pd.DataFrame) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Extract regression features and target values.

    Args:
        frame: A Sephora pricing dataframe.

    Returns:
        A tuple of the feature dataframe and the target array.
    """
    features = frame[list(REGRESSION_FEATURES)]
    target = frame[PRICING_TARGET].to_numpy(dtype=float)
    return features, target


def main() -> int:
    """Train, evaluate, and persist all traditional ML baselines.

    Returns:
        The process exit code.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cls_train = _load(ARTIFACTS_DIR / "skincare_train.parquet")
    cls_val = _load(ARTIFACTS_DIR / "skincare_val.parquet")
    cls_test = _load(ARTIFACTS_DIR / "skincare_test.parquet")
    reg_train = _load(ARTIFACTS_DIR / "pricing_train.parquet")
    reg_val = _load(ARTIFACTS_DIR / "pricing_val.parquet")
    reg_test = _load(ARTIFACTS_DIR / "pricing_test.parquet")

    x_cls_train, y_cls_train = _prepare_classification(cls_train)
    x_cls_val, y_cls_val = _prepare_classification(cls_val)
    x_cls_test, y_cls_test = _prepare_classification(cls_test)

    x_reg_train, y_reg_train = _prepare_regression(reg_train)
    x_reg_val, y_reg_val = _prepare_regression(reg_val)
    x_reg_test, y_reg_test = _prepare_regression(reg_test)

    classification = build_classification_models(DETERMINISTIC_SEED)
    regression = build_regression_models(DETERMINISTIC_SEED)

    report: dict[str, dict[str, dict[str, float]]] = {
        "classification": {},
        "regression": {},
    }

    for name in MODEL_NAMES:
        classification[name].fit(x_cls_train, y_cls_train)
        joblib.dump(classification[name], MODELS_DIR / model_filename("classification", name))
        report["classification"][name] = {
            "val_accuracy": _accuracy(y_cls_val, classification[name].predict(x_cls_val)),
            "test_accuracy": _accuracy(y_cls_test, classification[name].predict(x_cls_test)),
        }

        regression[name].fit(x_reg_train, y_reg_train)
        joblib.dump(regression[name], MODELS_DIR / model_filename("regression", name))
        report["regression"][name] = {
            "val_mape": _mape(y_reg_val, regression[name].predict(x_reg_val)),
            "test_mape": _mape(y_reg_test, regression[name].predict(x_reg_test)),
            "test_rmse": _rmse(y_reg_test, regression[name].predict(x_reg_test)),
        }

    metrics_path = EVALUATION_RESULTS_DIR / "ml_baseline_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Classification (test accuracy) ===")
    for name, metrics in report["classification"].items():
        print(f"  {name:18s} {metrics['test_accuracy']:.4f}")
    print("=== Regression (test MAPE %) ===")
    for name, metrics in report["regression"].items():
        print(f"  {name:18s} {metrics['test_mape']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
