"""Traditional ML baseline models for classification and regression.

Defines the five-model baseline roster (Random Forest, Gradient Boosting,
XGBoost, CatBoost, and a stacking meta-ensemble) together with an inference
service that loads persisted pipelines from ``src/artifacts/models/``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

CLASS_LABELS: tuple[str, ...] = ("Holy_Grail_Safe", "Neutral", "Irritation_Risk")
CLASS_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(CLASS_LABELS)}
INDEX_TO_CLASS: dict[int, str] = {i: label for i, label in enumerate(CLASS_LABELS)}

CLASSIFICATION_FEATURES: tuple[str, ...] = (
    "brand_name",
    "product_category",
    "primary_active",
    "has_fragrance",
    "has_alcohol_denat",
    "comedogenic_rating",
    "skin_type",
    "skin_tone",
)
CLASSIFICATION_CATEGORICAL: tuple[str, ...] = (
    "brand_name",
    "product_category",
    "primary_active",
    "skin_type",
    "skin_tone",
)

REGRESSION_FEATURES: tuple[str, ...] = (
    "brand_name",
    "product_category",
    "size_ml",
    "active_ingredient_count",
    "concentration_rank",
    "loves_count",
    "reviews_count",
    "average_rating",
    "sephora_exclusive",
    "clean_at_sephora",
)
REGRESSION_CATEGORICAL: tuple[str, ...] = ("brand_name", "product_category")

MODEL_NAMES: tuple[str, ...] = (
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "catboost",
    "stacking",
)


def _to_float64(frame: object) -> NDArray[np.float64]:
    """Coerce a dataframe or array to a float64 numpy array.

    Args:
        frame: The numeric columns to coerce.

    Returns:
        A float64 numpy array.
    """
    return np.asarray(frame, dtype=np.float64)


def _build_pipeline(
    estimator: object,
    features: tuple[str, ...],
    categorical: tuple[str, ...],
) -> Pipeline:
    """Wrap an estimator with a shared one-hot/numeric preprocessing pipeline.

    Args:
        estimator: The fitted-scikit-compatible estimator.
        features: The ordered feature columns.
        categorical: The subset of feature columns that are categorical.

    Returns:
        A pipeline that one-hot encodes categoricals and passes numerics through.
    """
    numeric = [column for column in features if column not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical),
            ),
            (
                "num",
                FunctionTransformer(_to_float64, validate=False),
                numeric,
            ),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def build_classification_models(seed: int) -> dict[str, Pipeline]:
    """Build the five classification baselines.

    Args:
        seed: The deterministic random seed.

    Returns:
        A mapping from model name to a fitted-capable pipeline.
    """
    estimators: dict[str, object] = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=seed,
            eval_metric="mlogloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=100, learning_rate=0.1, random_seed=seed, verbose=0
        ),
        "stacking": StackingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(n_estimators=100, random_state=seed, eval_metric="mlogloss"),
                ),
                (
                    "cat",
                    CatBoostClassifier(iterations=100, random_seed=seed, verbose=0),
                ),
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=seed),
            cv=3,
            n_jobs=-1,
        ),
    }
    return {
        name: _build_pipeline(estimator, CLASSIFICATION_FEATURES, CLASSIFICATION_CATEGORICAL)
        for name, estimator in estimators.items()
    }


def build_regression_models(seed: int) -> dict[str, Pipeline]:
    """Build the five regression baselines.

    Args:
        seed: The deterministic random seed.

    Returns:
        A mapping from model name to a fitted-capable pipeline.
    """
    estimators: dict[str, object] = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=seed
        ),
        "xgboost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=seed),
        "catboost": CatBoostRegressor(
            iterations=100, learning_rate=0.1, random_seed=seed, verbose=0
        ),
        "stacking": StackingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
                ("xgb", XGBRegressor(n_estimators=100, random_state=seed)),
                ("cat", CatBoostRegressor(iterations=100, random_seed=seed, verbose=0)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1,
        ),
    }
    return {
        name: _build_pipeline(estimator, REGRESSION_FEATURES, REGRESSION_CATEGORICAL)
        for name, estimator in estimators.items()
    }


def model_filename(task: str, name: str) -> str:
    """Return the persisted filename for a model.

    Args:
        task: Either ``"classification"`` or ``"regression"``.
        name: The canonical model name.

    Returns:
        The filename (without directory).
    """
    return f"{task}_{name}.joblib"


class MLBaselineService:
    """Loads and serves the persisted traditional ML baseline models."""

    def __init__(self, models_dir: Path) -> None:
        """Instantiate the service.

        Args:
            models_dir: The directory containing persisted model pipelines.
        """
        self._models_dir = models_dir
        self._classification: dict[str, Pipeline] = {}
        self._regression: dict[str, Pipeline] = {}

    def load(self) -> None:
        """Load every persisted model pipeline from disk."""
        for name in MODEL_NAMES:
            self._classification[name] = joblib.load(
                self._models_dir / model_filename("classification", name)
            )
            self._regression[name] = joblib.load(
                self._models_dir / model_filename("regression", name)
            )

    def predict_classification(self, model_name: str, features: pd.DataFrame) -> str:
        """Predict the compatibility tier for a single product/skin profile.

        Args:
            model_name: The canonical model name.
            features: A one-row dataframe of classification features.

        Returns:
            The predicted compatibility tier label.
        """
        encoded = self._classification[model_name].predict(features[list(CLASSIFICATION_FEATURES)])
        return INDEX_TO_CLASS[int(np.ravel(encoded)[0])]

    def predict_regression(self, model_name: str, features: pd.DataFrame) -> float:
        """Predict the price in USD for a single product.

        Args:
            model_name: The canonical model name.
            features: A one-row dataframe of regression features.

        Returns:
            The predicted price in US dollars.
        """
        predicted = self._regression[model_name].predict(features[list(REGRESSION_FEATURES)])
        return float(np.ravel(predicted)[0])
