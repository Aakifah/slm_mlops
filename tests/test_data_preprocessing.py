"""Deterministic data preparation pipeline tests.

All tests run offline against the seeded synthetic generator so the suite never
depends on network access or a live Ollama instance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_preprocessing.preprocessing.parsers.ingredients import (
    active_ingredient_count,
    comedogenic_rating,
    detect_actives,
    has_alcohol_denat,
    has_fragrance,
    parse_ingredient_text,
    parse_size_ml,
    primary_active,
)
from src.data_preprocessing.preprocessing.parsers.skincare import (
    compatibility_score,
    compatibility_tier,
)
from src.data_preprocessing.preprocessing.parsers.source import ingest
from src.data_preprocessing.preprocessing.parsers.synthetic import generate_synthetic
from src.data_preprocessing.preprocessing.processors.cleaner import deduplicate_exact
from src.data_preprocessing.preprocessing.processors.scaler import fit_scaler
from src.data_preprocessing.preprocessing.processors.splitter import (
    stratified_split,
    verify_no_cross_contamination,
)
from src.data_preprocessing.preprocessing.schemas import (
    PRICING_COLUMNS,
    PRICING_NUMERIC_COLUMNS,
    PRICING_TARGET,
    SKINCARE_COLUMNS,
)

_PRODUCT_COLUMN: str = "product_id"


def test_parse_ingredient_text_handles_missing_values() -> None:
    """Missing and NaN ingredient fields must map to an empty string."""
    assert parse_ingredient_text(None) == ""
    assert parse_ingredient_text(float("nan")) == ""


def test_parse_ingredient_text_flattens_stringified_list() -> None:
    """A stringified Python list must be flattened and lowercased."""
    raw = "['Water', 'Glycerin', 'Niacinamide']"
    assert parse_ingredient_text(raw) == "water glycerin niacinamide"


def test_detect_actives_and_primary() -> None:
    """Active detection must map synonyms to canonical ingredient names."""
    text = "water niacinamide retinol salicylic acid"
    assert detect_actives(text) == ["Niacinamide", "Retinol", "Salicylic_Acid"]
    assert primary_active(text) == "Niacinamide"
    assert active_ingredient_count(text) == 3


def test_fragrance_and_alcohol_detection() -> None:
    """Fragrance and denatured alcohol keywords must be detected."""
    text = "water fragrance alcohol denat"
    assert has_fragrance(text)
    assert has_alcohol_denat(text)


def test_comedogenic_rating_lookup() -> None:
    """Comedogenic rating must reflect the maximum matched ingredient."""
    assert comedogenic_rating("water glycerin") == 0
    assert comedogenic_rating("water coconut oil") == 4
    assert comedogenic_rating("isopropyl myristate coconut oil") == 5


def test_parse_size_ml_handles_leading_decimal() -> None:
    """Leading-decimal sizes must not be mis-parsed as large quantities."""
    assert parse_size_ml(".0176 oz") == pytest.approx(0.52, abs=0.01)
    assert parse_size_ml("100 mL") == pytest.approx(100.0)
    assert parse_size_ml("1 oz") == pytest.approx(29.57, abs=0.01)


def test_compatibility_tiers_are_valid() -> None:
    """Score thresholds must map into the three valid tier labels."""
    valid = {"Holy_Grail_Safe", "Neutral", "Irritation_Risk"}
    assert compatibility_tier(5) in valid
    assert compatibility_tier(2) == "Neutral"
    assert compatibility_tier(-3) == "Irritation_Risk"
    assert compatibility_score("Niacinamide", False, False, 0, "sensitive", "fair") == 3


def test_synthetic_generation_is_deterministic() -> None:
    """Two runs with the same seed must produce identical frames."""
    pricing_a, skincare_a = generate_synthetic(seed=7, n_pricing=200, n_skincare=50)
    pricing_b, skincare_b = generate_synthetic(seed=7, n_pricing=200, n_skincare=50)
    assert pricing_a.equals(pricing_b)
    assert skincare_a.equals(skincare_b)


def test_offline_ingestion_conforms_to_schemas() -> None:
    """Offline ingestion must return schema-conformant non-empty frames."""
    data = ingest(offline=True)
    assert data.source == "synthetic"
    assert list(data.pricing.columns) == list(PRICING_COLUMNS)
    assert list(data.skincare.columns) == list(SKINCARE_COLUMNS)
    assert not data.pricing.empty
    assert not data.skincare.empty


def test_split_proportions_and_no_contamination() -> None:
    """Splits must approximate 70/15/15 with zero product overlap."""
    pricing, _ = generate_synthetic(seed=3, n_pricing=600, n_skincare=150)
    train, val, test = stratified_split(pricing, PRICING_TARGET, _PRODUCT_COLUMN, seed=3)
    total = len(pricing)
    assert abs(len(train) / total - 0.70) < 0.03
    assert abs(len(val) / total - 0.15) < 0.03
    assert abs(len(test) / total - 0.15) < 0.03
    verify_no_cross_contamination(train, val, test, _PRODUCT_COLUMN)


def test_cross_contamination_raises_on_overlap() -> None:
    """Overlapping products across splits must raise a ValueError."""
    pricing, _ = generate_synthetic(seed=3, n_pricing=200, n_skincare=50)
    train, val, test = stratified_split(pricing, PRICING_TARGET, _PRODUCT_COLUMN, seed=3)
    contaminated_val = pd.concat([val, train.head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        verify_no_cross_contamination(train, contaminated_val, test, _PRODUCT_COLUMN)


def test_scaler_bounds_are_strictly_increasing() -> None:
    """Fitted scaler bounds must have min < max for every numeric feature."""
    pricing, _ = generate_synthetic(seed=4, n_pricing=400, n_skincare=100)
    train, _, _ = stratified_split(pricing, PRICING_TARGET, _PRODUCT_COLUMN, seed=4)
    scaler = fit_scaler(train, PRICING_NUMERIC_COLUMNS, "sephora_pricing")
    assert set(scaler.features) == set(PRICING_NUMERIC_COLUMNS)
    for bounds in scaler.features.values():
        assert bounds.min < bounds.max


def test_deduplicate_exact_removes_duplicates() -> None:
    """Exact duplicate rows must be collapsed to a single row."""
    pricing, _ = generate_synthetic(seed=5, n_pricing=100, n_skincare=25)
    duplicated = pricing.sample(10, random_state=1)
    frame_with_dupes = pd.concat([pricing, duplicated], ignore_index=True)
    deduped = deduplicate_exact(frame_with_dupes)
    assert len(deduped) == len(pricing)
