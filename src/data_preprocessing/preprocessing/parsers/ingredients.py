"""Deterministic INCI ingredient parsing and dermatological feature extraction.

The raw Sephora ``ingredients`` field is a stringified Python list whose
elements may carry a ``"Product Name:"`` prefix. These helpers normalize that
representation into a single lowercase text blob and extract the structured
features consumed by both the classification and regression pipelines.
"""

from __future__ import annotations

import ast
import math
import re

from .constants import (
    ACTIVE_INGREDIENT_KEYWORDS,
    ALCOHOL_DENAT_KEYWORDS,
    COMEDOGENIC_LOOKUP,
    FRAGRANCE_KEYWORDS,
    NON_CLEAN_KEYWORDS,
)

_ML_PATTERN: re.Pattern[str] = re.compile(r"(\d*\.?\d+)\s*ml", re.IGNORECASE)
_OZ_PATTERN: re.Pattern[str] = re.compile(r"(\d*\.?\d+)\s*oz", re.IGNORECASE)
_GRAM_PATTERN: re.Pattern[str] = re.compile(r"(\d*\.?\d+)\s*g\b", re.IGNORECASE)

_OZ_TO_ML: float = 29.5735


def parse_ingredient_text(raw: object) -> str:
    """Normalize a raw ingredient field into a single lowercase searchable text.

    Args:
        raw: The raw stringified ingredient list, ``None``, or ``NaN``.

    Returns:
        The joined lowercase ingredient text, or an empty string when missing.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and math.isnan(raw):
        return ""
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            return " ".join(_strip_name_prefix(item) for item in parsed).lower()
    return text.lower()


def _strip_name_prefix(element: object) -> str:
    """Remove a leading ``"Name:"`` prefix from a list element when present.

    Args:
        element: A single item from a parsed ingredient list.

    Returns:
        The element with any short name prefix stripped.
    """
    text = str(element)
    index = text.find(": ")
    if 0 < index < 40:
        return text[index + 2 :]
    return text


def detect_actives(text: str) -> list[str]:
    """Return the canonical active ingredients present in ``text``.

    Args:
        text: The normalized lowercase ingredient text.

    Returns:
        Canonical active names in priority order.
    """
    return [
        active
        for active, keywords in ACTIVE_INGREDIENT_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]


def primary_active(text: str) -> str:
    """Return the single dominant active ingredient.

    When no recognized active is present the neutral baseline ``Niacinamide``
    is returned so that every product receives a deterministic category.

    Args:
        text: The normalized lowercase ingredient text.

    Returns:
        The canonical name of the primary active ingredient.
    """
    actives = detect_actives(text)
    return actives[0] if actives else "Niacinamide"


def active_ingredient_count(text: str) -> int:
    """Count the number of distinct recognized active ingredients.

    Args:
        text: The normalized lowercase ingredient text.

    Returns:
        The count of recognized actives present.
    """
    return len(detect_actives(text))


def has_fragrance(text: str) -> bool:
    """Detect whether the ingredient list contains fragrance or parfum.

    Args:
        text: The normalized lowercase ingredient text.

    Returns:
        ``True`` when a fragrance keyword is present.
    """
    return any(keyword in text for keyword in FRAGRANCE_KEYWORDS)


def has_alcohol_denat(text: str) -> bool:
    """Detect whether the ingredient list contains denatured alcohol.

    Args:
        text: The normalized lowercase ingredient text.

    Returns:
        ``True`` when a denatured-alcohol keyword is present.
    """
    return any(keyword in text for keyword in ALCOHOL_DENAT_KEYWORDS)


def comedogenic_rating(text: str) -> int:
    """Return the maximum comedogenic rating across detected ingredients.

    Args:
        text: The normalized lowercase ingredient text.

    Returns:
        An integer rating in the inclusive range ``0..5``.
    """
    ratings = [rating for ingredient, rating in COMEDOGENIC_LOOKUP.items() if ingredient in text]
    return max(ratings) if ratings else 0


def is_clean(text: str, has_fragrance_flag: bool, has_alcohol_flag: bool) -> bool:
    """Determine whether a formulation qualifies as a clean product.

    Args:
        text: The normalized lowercase ingredient text.
        has_fragrance_flag: Whether fragrance was detected.
        has_alcohol_flag: Whether denatured alcohol was detected.

    Returns:
        ``True`` when the product avoids fragrance, alcohol, and known
        non-clean preservative families.
    """
    if has_fragrance_flag or has_alcohol_flag:
        return False
    return not any(keyword in text for keyword in NON_CLEAN_KEYWORDS)


def parse_size_ml(raw: object) -> float | None:
    """Parse a free-text product size into a millilitre quantity.

    Args:
        raw: The raw ``size`` string, ``None``, or ``NaN``.

    Returns:
        The size in millilitres, or ``None`` when no quantity can be parsed.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    text = str(raw).lower()
    match = _ML_PATTERN.search(text)
    if match:
        return float(match.group(1))
    match = _OZ_PATTERN.search(text)
    if match:
        return round(float(match.group(1)) * _OZ_TO_ML, 2)
    match = _GRAM_PATTERN.search(text)
    if match:
        return float(match.group(1))
    return None
