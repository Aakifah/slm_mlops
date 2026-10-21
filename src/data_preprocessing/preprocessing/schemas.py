"""Canonical dataset schemas shared by parsers, processors, and tests.

These tuples define the exact column order and the numeric feature set for the
two engineered datasets so that every producer and consumer of the Parquet
artifacts agrees on a single contract.
"""

from __future__ import annotations

from typing import Final

PRICING_COLUMNS: Final[tuple[str, ...]] = (
    "product_id",
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
    "price_usd",
)

PRICING_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "size_ml",
    "active_ingredient_count",
    "concentration_rank",
    "loves_count",
    "reviews_count",
    "average_rating",
    "price_usd",
)

PRICING_TARGET: Final[str] = "price_usd"

SKINCARE_COLUMNS: Final[tuple[str, ...]] = (
    "product_id",
    "brand_name",
    "product_category",
    "primary_active",
    "has_fragrance",
    "has_alcohol_denat",
    "comedogenic_rating",
    "skin_type",
    "skin_tone",
    "compatibility_tier",
)

SKINCARE_TARGET: Final[str] = "compatibility_tier"
