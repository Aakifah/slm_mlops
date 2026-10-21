"""Primary ingestion for the Sephora pricing regression dataset.

Maps the raw Sephora product catalogue into the ``sephora_pricing`` feature
schema, applying deterministic domain engineering to derive ``size_ml``,
``active_ingredient_count``, ``concentration_rank``, and ``clean_at_sephora``
from the raw free-text attributes and INCI ingredient lists.
"""

from __future__ import annotations

import pandas as pd

from ..schemas import PRICING_COLUMNS
from .constants import PRICING_CATEGORIES
from .ingredients import (
    active_ingredient_count,
    has_alcohol_denat,
    has_fragrance,
    is_clean,
    parse_ingredient_text,
    parse_size_ml,
)


def parse_sephora_pricing(raw: pd.DataFrame) -> pd.DataFrame:
    """Transform raw Sephora products into the pricing feature dataframe.

    Args:
        raw: The raw Sephora product catalogue dataframe.

    Returns:
        A dataframe conforming to ``PRICING_COLUMNS`` with real-valued units.
    """
    frame = raw[raw["primary_category"].isin(PRICING_CATEGORIES)].copy()

    texts = frame["ingredients"].map(parse_ingredient_text)
    fragrance = texts.map(has_fragrance)
    alcohol = texts.map(has_alcohol_denat)

    frame["product_id"] = frame["product_id"].astype(str)
    frame["brand_name"] = frame["brand_name"].astype(str)
    frame["product_category"] = frame["primary_category"].astype(str)
    frame["size_ml"] = frame["size"].map(parse_size_ml)
    frame["active_ingredient_count"] = texts.map(active_ingredient_count).astype(int)
    frame["loves_count"] = pd.to_numeric(frame["loves_count"], errors="coerce").astype(int)
    frame["reviews_count"] = pd.to_numeric(frame["reviews"], errors="coerce").fillna(0).astype(int)
    frame["average_rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["sephora_exclusive"] = (
        pd.to_numeric(frame["sephora_exclusive"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )
    frame["clean_at_sephora"] = [
        is_clean(text, frag, alc) for text, frag, alc in zip(texts, fragrance, alcohol)
    ]
    frame["price_usd"] = pd.to_numeric(frame["price_usd"], errors="coerce")

    frame = frame.dropna(subset=["product_id", "brand_name", "size_ml", "price_usd"])

    frame["average_rating"] = frame["average_rating"].fillna(
        frame.groupby("product_category")["average_rating"].transform("median")
    )
    frame["average_rating"] = frame["average_rating"].fillna(frame["average_rating"].median())

    frame["concentration_rank"] = (
        frame.groupby("product_category")["active_ingredient_count"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    result = frame[list(PRICING_COLUMNS)].copy()
    result["size_ml"] = result["size_ml"].astype(float)
    result["average_rating"] = result["average_rating"].astype(float)
    result["price_usd"] = result["price_usd"].astype(float)
    return result.reset_index(drop=True)
