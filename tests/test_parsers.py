"""Primary parser tests against a deterministic raw Sephora fixture."""

from __future__ import annotations

import pandas as pd

from src.data_preprocessing.preprocessing.parsers.sephora import parse_sephora_pricing
from src.data_preprocessing.preprocessing.parsers.skincare import (
    derive_skincare_compatibility,
)
from src.data_preprocessing.preprocessing.schemas import (
    PRICING_COLUMNS,
    SKINCARE_COLUMNS,
)
from src.data_preprocessing.preprocessing.parsers.constants import (
    COMPATIBILITY_TIERS,
    PRICING_CATEGORIES,
    SKINCARE_CATEGORIES,
)


def _raw_fixture() -> pd.DataFrame:
    """Return a small deterministic raw Sephora catalogue dataframe."""
    return pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "product_name": [
                "Niacinamide 10% Serum",
                "Salicylic Acid Cleanser",
                "Hydrating Moisturizer",
                "Unseen Sunscreen SPF 40",
                "Eau de Parfum",
                "Matte Lipstick",
            ],
            "brand_name": [
                "The Ordinary",
                "Paula's Choice",
                "CeraVe",
                "Supergoop!",
                "Chanel",
                "NARS",
            ],
            "primary_category": [
                "Skincare",
                "Skincare",
                "Skincare",
                "Skincare",
                "Fragrance",
                "Makeup",
            ],
            "secondary_category": [
                "Treatments",
                "Cleansers",
                "Moisturizers",
                "Sunscreen",
                "Women",
                "Lip",
            ],
            "size": [
                "1 oz/ 30 mL",
                "6 oz / 180 mL",
                "1.7 oz / 50 mL",
                "0.25 oz/ 7.5 mL",
                "3.4 oz/ 100 mL",
                "0.1 oz/ 3 g",
            ],
            "ingredients": [
                "['Water', 'Niacinamide', 'Zinc PCA']",
                "['Water', 'Salicylic Acid', 'Fragrance', 'Alcohol Denat.']",
                "['Water', 'Glycerin', 'Hyaluronic Acid', 'Ceramide NP']",
                "['Water', 'Zinc Oxide']",
                "['Alcohol Denat. (SD Alcohol 39C), Parfum (Fragrance)']",
                "['Water', 'Dimethicone']",
            ],
            "loves_count": [1000, 500, 800, 2000, 300, 150],
            "reviews": [100, 50, 80, 200, 30, 15],
            "rating": [4.5, 4.0, 4.8, 4.2, 3.5, 4.0],
            "sephora_exclusive": [0, 0, 0, 1, 1, 0],
            "price_usd": [10.0, 30.0, 15.0, 34.0, 85.0, 32.0],
        }
    )


def test_parse_sephora_pricing_maps_features() -> None:
    """Pricing parsing must map raw attributes into the pricing schema."""
    pricing = parse_sephora_pricing(_raw_fixture())
    assert list(pricing.columns) == list(PRICING_COLUMNS)
    assert len(pricing) == 6
    assert set(pricing["product_category"]) <= set(PRICING_CATEGORIES)
    assert pricing["price_usd"].isna().sum() == 0
    assert pricing["size_ml"].isna().sum() == 0
    assert pricing.loc[0, "size_ml"] == 30.0
    assert pricing.loc[0, "active_ingredient_count"] == 1
    assert pricing.loc[2, "active_ingredient_count"] == 2
    assert bool(pricing.loc[1, "clean_at_sephora"]) is False


def test_derive_skincare_compatibility_expands_profiles() -> None:
    """Compatibility derivation must expand each skincare product across skin types."""
    skincare = derive_skincare_compatibility(_raw_fixture())
    assert list(skincare.columns) == list(SKINCARE_COLUMNS)
    assert len(skincare) == 16
    assert set(skincare["product_category"]) <= set(SKINCARE_CATEGORIES)
    assert set(skincare["compatibility_tier"]) <= set(COMPATIBILITY_TIERS)
    assert set(skincare["skin_type"]) == {"oily", "dry", "combination", "sensitive"}
