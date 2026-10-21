"""Deterministic synthetic fallback for offline ingestion.

When real data is unavailable (offline CI runs or no network), these generators
produce domain-realistic Cosmetics & Skincare datasets that conform to the exact
same schemas as the primary parsers. A fixed seed guarantees reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import Generator

from ..schemas import PRICING_COLUMNS, SKINCARE_COLUMNS
from .constants import (
    ACTIVE_INGREDIENT_KEYWORDS,
    PRICING_CATEGORIES,
    REAL_BRANDS,
    SKIN_TONES,
    SKIN_TYPES,
    SKINCARE_CATEGORIES,
)
from .skincare import compatibility_score, compatibility_tier

_PRICING_PRICE_PARAMS: dict[str, tuple[float, float]] = {
    "Skincare": (45.0, 0.5),
    "Makeup": (35.0, 0.5),
    "Hair": (28.0, 0.5),
    "Fragrance": (85.0, 0.6),
    "Bath & Body": (20.0, 0.5),
}

_PRICING_SIZE_RANGES: dict[str, tuple[float, float]] = {
    "Skincare": (30.0, 200.0),
    "Makeup": (5.0, 50.0),
    "Hair": (50.0, 500.0),
    "Fragrance": (30.0, 100.0),
    "Bath & Body": (100.0, 500.0),
}

_ACTIVE_WEIGHTS: dict[str, float] = {
    "Niacinamide": 0.24,
    "Hyaluronic_Acid": 0.20,
    "Ceramides": 0.12,
    "Vitamin_C": 0.12,
    "Salicylic_Acid": 0.10,
    "Retinol": 0.08,
    "Glycolic_Acid": 0.08,
    "Azelaic_Acid": 0.06,
}


def generate_synthetic_pricing(rng: Generator, n_products: int = 6000) -> pd.DataFrame:
    """Generate a deterministic synthetic Sephora pricing dataset.

    Args:
        rng: A seeded NumPy random generator.
        n_products: The number of products to synthesize.

    Returns:
        A dataframe conforming to ``PRICING_COLUMNS``.
    """
    categories = list(PRICING_CATEGORIES)
    brands = list(REAL_BRANDS)
    n = n_products

    category = np.array([categories[i] for i in rng.integers(0, len(categories), n)])
    price = np.array(
        [
            rng.lognormal(
                mean=np.log(_PRICING_PRICE_PARAMS[cat][0]),
                sigma=_PRICING_PRICE_PARAMS[cat][1],
            )
            for cat in category
        ]
    )
    size = np.array([rng.uniform(*_PRICING_SIZE_RANGES[cat]) for cat in category])
    active_count = np.clip(rng.poisson(2.0, n), 0, 8)
    loves = rng.lognormal(mean=np.log(500.0), sigma=1.5, size=n).astype(int)
    reviews = np.maximum(loves * rng.uniform(0.2, 0.9, n), 0).astype(int)
    rating = np.clip(rng.normal(4.2, 0.4, n), 1.0, 5.0)

    frame = pd.DataFrame(
        {
            "product_id": [f"SYN-{i:05d}" for i in range(n)],
            "brand_name": [brands[i] for i in rng.integers(0, len(brands), n)],
            "product_category": category,
            "size_ml": np.round(size, 2),
            "active_ingredient_count": active_count,
            "loves_count": loves,
            "reviews_count": reviews,
            "average_rating": np.round(rating, 3),
            "sephora_exclusive": rng.integers(0, 2, n).astype(bool),
            "clean_at_sephora": rng.integers(0, 2, n).astype(bool),
            "price_usd": np.round(price, 2),
        }
    )
    frame["concentration_rank"] = (
        frame.groupby("product_category")["active_ingredient_count"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return frame[list(PRICING_COLUMNS)].reset_index(drop=True)


def _sample_primary_active(rng: Generator) -> str:
    """Sample a primary active according to realistic prevalence weights.

    Args:
        rng: A seeded NumPy random generator.

    Returns:
        A canonical active ingredient name.
    """
    actives = list(ACTIVE_INGREDIENT_KEYWORDS.keys())
    weights = [_ACTIVE_WEIGHTS[active] for active in actives]
    return str(rng.choice(actives, p=weights))


def generate_synthetic_skincare(rng: Generator, n_products: int = 1500) -> pd.DataFrame:
    """Generate a deterministic synthetic skincare compatibility dataset.

    Args:
        rng: A seeded NumPy random generator.
        n_products: The number of products to synthesize (expanded x4 skin types).

    Returns:
        A dataframe conforming to ``SKINCARE_COLUMNS``.
    """
    brands = list(REAL_BRANDS)
    categories = list(SKINCARE_CATEGORIES)
    tones = list(SKIN_TONES)

    comedogenic_choices = np.array([0, 1, 2, 3, 4, 5])
    comedogenic_probabilities = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])

    rows: list[dict[str, object]] = []
    for i in range(n_products):
        product_id = f"SYN-{i:05d}"
        brand_name = brands[int(rng.integers(0, len(brands)))]
        category = categories[int(rng.integers(0, len(categories)))]
        primary = _sample_primary_active(rng)
        fragrance = bool(rng.random() < 0.40)
        alcohol = bool(rng.random() < 0.15)
        comedogenic = int(rng.choice(comedogenic_choices, p=comedogenic_probabilities))
        for skin_type in SKIN_TYPES:
            skin_tone = tones[int(rng.integers(0, len(tones)))]
            score = compatibility_score(
                primary, fragrance, alcohol, comedogenic, skin_type, skin_tone
            )
            rows.append(
                {
                    "product_id": product_id,
                    "brand_name": brand_name,
                    "product_category": category,
                    "primary_active": primary,
                    "has_fragrance": fragrance,
                    "has_alcohol_denat": alcohol,
                    "comedogenic_rating": comedogenic,
                    "skin_type": skin_type,
                    "skin_tone": skin_tone,
                    "compatibility_tier": compatibility_tier(score),
                }
            )

    frame = pd.DataFrame(rows, columns=SKINCARE_COLUMNS)
    frame["has_fragrance"] = frame["has_fragrance"].astype(bool)
    frame["has_alcohol_denat"] = frame["has_alcohol_denat"].astype(bool)
    frame["comedogenic_rating"] = frame["comedogenic_rating"].astype(int)
    return frame.reset_index(drop=True)


def generate_synthetic(
    seed: int, n_pricing: int = 6000, n_skincare: int = 1500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both synthetic datasets from a single seed.

    Args:
        seed: The deterministic random seed.
        n_pricing: The number of pricing products to synthesize.
        n_skincare: The number of skincare products to synthesize.

    Returns:
        A tuple of ``(pricing_frame, skincare_frame)``.
    """
    rng = np.random.default_rng(seed)
    pricing = generate_synthetic_pricing(rng, n_pricing)
    skincare = generate_synthetic_skincare(rng, n_skincare)
    return pricing, skincare
