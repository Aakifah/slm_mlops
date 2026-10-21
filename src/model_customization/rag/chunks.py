"""Deterministic construction of schema-metadata RAG chunks.

These chunks describe every column of both datasets together with the natural
language aliases used to resolve user questions to real schema columns. The
output is validated by ``RAGChunkSchema`` before being persisted to disk.
"""

from __future__ import annotations

from src.backend.schemas.api import RAGChunk, RAGChunkSchema

_PRICING_SPEC: dict[str, tuple[str, tuple[str, ...]]] = {
    "product_id": ("Unique product identifier.", ("id", "sku")),
    "brand_name": ("Beauty brand name.", ("brand", "maker", "label", "company")),
    "product_category": ("Product category.", ("category", "type", "kind")),
    "size_ml": ("Product size in millilitres.", ("size", "volume", "ml", "amount", "quantity")),
    "active_ingredient_count": (
        "Number of active ingredients in the formula.",
        ("active count", "actives", "ingredient count", "how many actives"),
    ),
    "concentration_rank": (
        "Rank of active ingredient concentration within its category.",
        ("concentration", "strength", "potency", "rank"),
    ),
    "loves_count": ("Number of user loves.", ("loves", "favorites", "likes", "hearts")),
    "reviews_count": ("Number of customer reviews.", ("reviews", "review count", "feedback")),
    "average_rating": (
        "Average product rating from one to five.",
        ("rating", "stars", "score", "quality"),
    ),
    "sephora_exclusive": (
        "Whether the product is exclusive to Sephora.",
        ("exclusive", "only at sephora"),
    ),
    "clean_at_sephora": (
        "Whether the product meets the clean standard.",
        ("clean", "clean beauty"),
    ),
    "price_usd": (
        "Product price in US dollars.",
        ("price", "cost", "usd", "dollar", "value", "how much"),
    ),
}

_SKINCARE_SPEC: dict[str, tuple[str, tuple[str, ...]]] = {
    "product_id": ("Unique product identifier.", ("id", "sku")),
    "brand_name": ("Beauty brand name.", ("brand", "maker", "label", "company")),
    "product_category": (
        "Skincare product category.",
        (
            "category",
            "type",
            "kind",
            "moisturizer",
            "serum",
            "cleanser",
            "sunscreen",
            "toner",
            "exfoliant",
        ),
    ),
    "primary_active": (
        "Primary active ingredient in the formula.",
        ("active", "key ingredient", "star ingredient", "actives", "ingredient"),
    ),
    "has_fragrance": (
        "Whether the formula contains fragrance.",
        ("fragrance", "scented", "perfume", "scent"),
    ),
    "has_alcohol_denat": (
        "Whether the formula contains denatured alcohol.",
        ("alcohol", "denatured alcohol", "drying alcohol", "sd alcohol"),
    ),
    "comedogenic_rating": (
        "Comedogenic rating from zero to five indicating pore-clogging potential.",
        ("comedogenic", "pore clogging", "pore-clogging", "acne", "breakout", "clog"),
    ),
    "skin_type": ("Skin type.", ("skin", "oily", "dry", "combination", "sensitive")),
    "skin_tone": ("Skin tone.", ("tone", "shade", "complexion")),
    "compatibility_tier": (
        "Compatibility tier indicating whether the product is safe for a skin profile.",
        ("compatibility", "safe", "irritation", "holy grail", "holy_grail_safe", "irritation risk"),
    ),
}

_DOMAIN_ALIAS_CHUNKS: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "sephora_pricing",
        "price_usd",
        ("dupe", "overpriced", "luxury tax", "affordable", "cheap", "expensive", "bargain"),
    ),
    (
        "skincare_compatibility",
        "comedogenic_rating",
        ("pore-clogging", "acne-safe", "non-comedogenic", "won't break me out", "breakout"),
    ),
    (
        "skincare_compatibility",
        "compatibility_tier",
        ("safe for my skin", "will this irritate", "holy grail", "irritation", "good for me"),
    ),
]


def _dataset_overview(dataset: str, columns: tuple[str, ...]) -> RAGChunk:
    """Build the dataset-level overview chunk.

    Args:
        dataset: The dataset name.
        columns: The ordered columns of the dataset.

    Returns:
        A validated ``RAGChunk`` describing the whole dataset.
    """
    text = f"{dataset} dataset with columns: {', '.join(columns)}"
    return RAGChunk(
        id=f"{dataset}.overview", dataset=dataset, text=text, columns=list(columns), aliases=[]
    )


def _column_chunk(
    dataset: str, column: str, description: str, aliases: tuple[str, ...]
) -> RAGChunk:
    """Build a single column-metadata chunk.

    Args:
        dataset: The dataset name.
        column: The column name.
        description: A natural-language description of the column.
        aliases: Natural-language aliases for the column.

    Returns:
        A validated ``RAGChunk`` for the column.
    """
    text = f"{column}: {description} Aliases: {', '.join(aliases)}"
    return RAGChunk(
        id=f"{dataset}.{column}",
        dataset=dataset,
        text=text,
        columns=[column],
        aliases=list(aliases),
    )


def build_rag_chunks() -> RAGChunkSchema:
    """Construct the full validated schema-metadata chunk set.

    Returns:
        A ``RAGChunkSchema`` covering both datasets and their domain aliases.
    """
    chunks: list[RAGChunk] = []

    pricing_columns = tuple(_PRICING_SPEC.keys())
    chunks.append(_dataset_overview("sephora_pricing", pricing_columns))
    for column, (description, aliases) in _PRICING_SPEC.items():
        chunks.append(_column_chunk("sephora_pricing", column, description, aliases))

    skincare_columns = tuple(_SKINCARE_SPEC.keys())
    chunks.append(_dataset_overview("skincare_compatibility", skincare_columns))
    for column, (description, aliases) in _SKINCARE_SPEC.items():
        chunks.append(_column_chunk("skincare_compatibility", column, description, aliases))

    for dataset, column, aliases in _DOMAIN_ALIAS_CHUNKS:
        spec = _PRICING_SPEC if dataset == "sephora_pricing" else _SKINCARE_SPEC
        description = spec[column][0]
        chunks.append(_column_chunk(dataset, column, description, aliases))

    return RAGChunkSchema(chunks=chunks)
