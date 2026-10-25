"""Traditional ML baseline model tests."""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path

from src.backend.services.ml_baseline import (
    CLASS_LABELS,
    CLASS_TO_INDEX,
    CLASSIFICATION_FEATURES,
    INDEX_TO_CLASS,
    MODEL_NAMES,
    REGRESSION_FEATURES,
    MLBaselineService,
    build_classification_models,
    build_regression_models,
    model_filename,
)
from src.data_preprocessing.preprocessing.parsers.synthetic import generate_synthetic
from src.data_preprocessing.preprocessing.schemas import PRICING_TARGET, SKINCARE_TARGET


def test_label_mapping_roundtrips() -> None:
    """Class labels must map to indices and back losslessly."""
    assert set(CLASS_TO_INDEX) == set(CLASS_LABELS)
    assert all(INDEX_TO_CLASS[CLASS_TO_INDEX[label]] == label for label in CLASS_LABELS)


def test_build_classification_models_returns_five() -> None:
    """The classification factory must produce all five named models."""
    models = build_classification_models(seed=0)
    assert set(models) == set(MODEL_NAMES)


def test_build_regression_models_returns_five() -> None:
    """The regression factory must produce all five named models."""
    models = build_regression_models(seed=0)
    assert set(models) == set(MODEL_NAMES)


def test_random_forest_classification_fits_and_predicts() -> None:
    """A classification pipeline must fit and emit valid encoded labels."""
    _, skincare = generate_synthetic(seed=1, n_pricing=40, n_skincare=40)
    model = build_classification_models(seed=1)["random_forest"]
    features = skincare[list(CLASSIFICATION_FEATURES)]
    labels = skincare[SKINCARE_TARGET].map(CLASS_TO_INDEX)
    model.fit(features, labels)
    predictions = np.ravel(model.predict(features))
    assert len(predictions) == len(labels)
    assert set(predictions) <= {0, 1, 2}


def test_random_forest_regression_fits_and_predicts() -> None:
    """A regression pipeline must fit and emit real-valued predictions."""
    pricing, _ = generate_synthetic(seed=1, n_pricing=60, n_skincare=40)
    model = build_regression_models(seed=1)["random_forest"]
    features = pricing[list(REGRESSION_FEATURES)]
    target = pricing[PRICING_TARGET].astype(float)
    model.fit(features, target)
    predictions = model.predict(features)
    assert predictions.shape[0] == len(target)


def test_model_pipeline_joblib_roundtrip(tmp_path: Path) -> None:
    """A fitted pipeline must survive a joblib save/load cycle."""
    _, skincare = generate_synthetic(seed=2, n_pricing=40, n_skincare=40)
    model = build_classification_models(seed=2)["random_forest"]
    features = skincare[list(CLASSIFICATION_FEATURES)]
    labels = skincare[SKINCARE_TARGET].map(CLASS_TO_INDEX)
    model.fit(features, labels)

    path = tmp_path / model_filename("classification", "random_forest")
    joblib.dump(model, path)
    reloaded = joblib.load(path)

    assert np.array_equal(model.predict(features), reloaded.predict(features))


def test_service_predicts_classification(tmp_path: Path) -> None:
    """The service must decode a prediction back into a tier label."""
    _, skincare = generate_synthetic(seed=3, n_pricing=40, n_skincare=40)
    model = build_classification_models(seed=3)["random_forest"]
    features = skincare[list(CLASSIFICATION_FEATURES)]
    labels = skincare[SKINCARE_TARGET].map(CLASS_TO_INDEX)
    model.fit(features, labels)

    service = MLBaselineService(tmp_path)
    service._classification["random_forest"] = model

    prediction = service.predict_classification("random_forest", features.iloc[[0]])
    assert prediction in CLASS_LABELS


def test_service_predicts_regression(tmp_path: Path) -> None:
    """The service must return a float price prediction."""
    pricing, _ = generate_synthetic(seed=4, n_pricing=60, n_skincare=40)
    model = build_regression_models(seed=4)["random_forest"]
    features = pricing[list(REGRESSION_FEATURES)]
    target = pricing[PRICING_TARGET].astype(float)
    model.fit(features, target)

    service = MLBaselineService(tmp_path)
    service._regression["random_forest"] = model

    prediction = service.predict_regression("random_forest", features.iloc[[0]])
    assert isinstance(prediction, float)
    assert prediction > 0
