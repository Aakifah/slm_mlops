"""Deterministic derivation of the skincare compatibility classification set.

Real Sephora skincare products are combined with a dermatological scoring model
to produce a ``(product x skin profile)`` compatibility matrix. Each row scores
whether a product is safe for a particular skin type and tone, yielding the
three-tier ``compatibility_tier`` target.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from ..schemas import SKINCARE_COLUMNS, SKINCARE_TARGET
from .constants import (
    COMPATIBILITY_TIERS,
    SKIN_TONES,
    SKIN_TYPES,
    SKINCARE_CATEGORIES,
)
from .ingredients import (
    comedogenic_rating,
    has_alcohol_denat,
    has_fragrance,
    parse_ingredient_text,
    primary_active,
)

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Sunscreen": ("sunscreen", "spf", "sun screen", "sun care", "uv defense"),
    "Cleanser": ("cleanser", "cleansing", "face wash", "micellar", "cleaning"),
    "Exfoliant": ("exfoliat", "peel", "scrub", "polish", "resurfacing"),
    "Toner": ("toner", "tonic", "essence", "mist"),
    "Serum": ("serum", "treatment", "oil", "ampoule", "concentrate", "booster"),
    "Moisturizer": ("moisturiz", "cream", "lotion", "balm", "hydration", "barrier"),
}

_SECONDARY_CATEGORY_MAP: dict[str, str] = {
    "Moisturizers": "Moisturizer",
    "Treatments": "Serum",
    "Cleansers": "Cleanser",
    "Sunscreen": "Sunscreen",
}

_ACTIVE_EFFECTS: dict[str, dict[str, int]] = {
    "Niacinamide": {"oily": 3, "dry": 3, "combination": 3, "sensitive": 3},
    "Hyaluronic_Acid": {"oily": 2, "dry": 3, "combination": 2, "sensitive": 3},
    "Ceramides": {"oily": 2, "dry": 3, "combination": 2, "sensitive": 3},
    "Azelaic_Acid": {"oily": 2, "dry": 2, "combination": 2, "sensitive": 2},
    "Vitamin_C": {"oily": 1, "dry": 0, "combination": 1, "sensitive": -1},
    "Salicylic_Acid": {"oily": 2, "dry": -2, "combination": 1, "sensitive": -2},
    "Retinol": {"oily": 1, "dry": 0, "combination": 1, "sensitive": -2},
    "Glycolic_Acid": {"oily": 1, "dry": -2, "combination": 0, "sensitive": -2},
}

_EXFOLIANT_ACTIVES: frozenset[str] = frozenset({"Salicylic_Acid", "Glycolic_Acid", "Retinol"})

_SAFE_THRESHOLD: int = 3
_NEUTRAL_THRESHOLD: int = 1


def classify_skincare_category(product_name: object, secondary_category: object) -> str:
    """Classify a skincare product into a canonical product category.

    Args:
        product_name: The raw product name.
        secondary_category: The raw secondary category as a fallback signal.

    Returns:
        The canonical category name, or ``"Other"`` when unmatched.
    """
    name = str(product_name).lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in name for keyword in keywords):
            return category
    return _SECONDARY_CATEGORY_MAP.get(str(secondary_category), "Other")


def _active_effect(active: str, skin_type: str) -> int:
    """Return the compatibility contribution of an active for a skin type.

    Args:
        active: The canonical primary active ingredient.
        skin_type: The target skin type.

    Returns:
        The signed compatibility contribution.
    """
    return _ACTIVE_EFFECTS.get(active, {}).get(skin_type, 0)


def compatibility_score(
    primary: str,
    has_fragrance_flag: bool,
    has_alcohol_flag: bool,
    comedogenic: int,
    skin_type: str,
    skin_tone: str,
) -> int:
    """Compute the deterministic compatibility score for a product/skin pair.

    Args:
        primary: The canonical primary active ingredient.
        has_fragrance_flag: Whether fragrance is present.
        has_alcohol_flag: Whether denatured alcohol is present.
        comedogenic: The comedogenic rating in ``0..5``.
        skin_type: The target skin type.
        skin_tone: The target skin tone.

    Returns:
        A signed integer score where higher values indicate safer use.
    """
    score = 0
    if has_fragrance_flag:
        score -= 2
    if has_alcohol_flag:
        score -= 1
    if skin_type == "oily":
        score -= comedogenic
    elif skin_type in {"combination", "sensitive"}:
        score -= comedogenic // 2
    score += _active_effect(primary, skin_type)
    if primary in _EXFOLIANT_ACTIVES and skin_tone in {"tan", "deep"}:
        score -= 1
    return score


def compatibility_tier(score: int) -> str:
    """Map a compatibility score to its three-tier label.

    Args:
        score: The compatibility score.

    Returns:
        One of the ``COMPATIBILITY_TIERS`` labels.
    """
    if score >= _SAFE_THRESHOLD:
        return COMPATIBILITY_TIERS[0]
    if score >= _NEUTRAL_THRESHOLD:
        return COMPATIBILITY_TIERS[1]
    return COMPATIBILITY_TIERS[2]


def _deterministic_skin_tone(product_id: object, skin_type: str) -> str:
    """Assign a stable skin tone for a product/skin-type pair.

    Args:
        product_id: The product identifier.
        skin_type: The target skin type.

    Returns:
        A deterministic skin tone from the supported set.
    """
    digest = hashlib.md5(f"{product_id}|{skin_type}".encode("utf-8")).digest()
    return SKIN_TONES[digest[0] % len(SKIN_TONES)]


def derive_skincare_compatibility(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the classification dataset from raw Sephora skincare products.

    Args:
        raw: The raw Sephora product dataframe.

    Returns:
        A dataframe conforming to ``SKINCARE_COLUMNS``.
    """
    skincare = raw[raw["primary_category"] == "Skincare"].copy()
    skincare = skincare[skincare["ingredients"].notna()].copy()
    skincare["category"] = [
        classify_skincare_category(name, secondary)
        for name, secondary in zip(skincare["product_name"], skincare["secondary_category"])
    ]
    skincare = skincare[skincare["category"].isin(SKINCARE_CATEGORIES)].copy()

    texts = skincare["ingredients"].map(parse_ingredient_text)
    skincare["primary_active"] = texts.map(primary_active)
    skincare["has_fragrance"] = texts.map(has_fragrance)
    skincare["has_alcohol_denat"] = texts.map(has_alcohol_denat)
    skincare["comedogenic_rating"] = texts.map(comedogenic_rating)

    products = skincare[
        [
            "product_id",
            "brand_name",
            "category",
            "primary_active",
            "has_fragrance",
            "has_alcohol_denat",
            "comedogenic_rating",
        ]
    ].drop_duplicates(subset=["product_id"])

    rows: list[dict[str, object]] = []
    for record in products.itertuples(index=False):
        for skin_type in SKIN_TYPES:
            skin_tone = _deterministic_skin_tone(record.product_id, skin_type)
            score = compatibility_score(
                record.primary_active,
                record.has_fragrance,
                record.has_alcohol_denat,
                record.comedogenic_rating,
                skin_type,
                skin_tone,
            )
            rows.append(
                {
                    "product_id": str(record.product_id),
                    "brand_name": str(record.brand_name),
                    "product_category": record.category,
                    "primary_active": record.primary_active,
                    "has_fragrance": record.has_fragrance,
                    "has_alcohol_denat": record.has_alcohol_denat,
                    "comedogenic_rating": record.comedogenic_rating,
                    "skin_type": skin_type,
                    "skin_tone": skin_tone,
                    SKINCARE_TARGET: compatibility_tier(score),
                }
            )

    frame = pd.DataFrame(rows, columns=SKINCARE_COLUMNS)
    frame["has_fragrance"] = frame["has_fragrance"].astype(bool)
    frame["has_alcohol_denat"] = frame["has_alcohol_denat"].astype(bool)
    frame["comedogenic_rating"] = frame["comedogenic_rating"].astype(int)
    return frame.reset_index(drop=True)
