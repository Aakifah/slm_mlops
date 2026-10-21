"""Tiered ingestion that selects between real and synthetic data sources.

Primary ingestion prefers a locally staged CSV or the credential-free Hugging
Face mirror of the Sephora catalogue. When the file is unavailable and network
access is disabled or fails, ingestion falls back to the deterministic synthetic
generator so the pipeline remains reproducible in offline CI environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..paths import (
    DETERMINISTIC_SEED,
    SEPHORA_FILENAME,
    SEPHORA_LOCAL_CSV,
    SEPHORA_REPO_ID,
)
from .sephora import parse_sephora_pricing
from .skincare import derive_skincare_compatibility
from .synthetic import generate_synthetic


@dataclass(frozen=True)
class IngestedData:
    """The result of ingestion with its provenance label."""

    pricing: pd.DataFrame
    skincare: pd.DataFrame
    source: str


def _load_raw_sephora(offline: bool) -> pd.DataFrame | None:
    """Load the raw Sephora catalogue from a local file or the Hub.

    Args:
        offline: When ``True``, force the synthetic fallback and skip all I/O.

    Returns:
        The raw catalogue dataframe, or ``None`` when unavailable.
    """
    if offline:
        return None
    if SEPHORA_LOCAL_CSV.exists():
        return pd.read_csv(SEPHORA_LOCAL_CSV)
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=SEPHORA_REPO_ID,
            filename=SEPHORA_FILENAME,
            repo_type="dataset",
        )
        return pd.read_csv(path)
    except Exception:
        return None


def ingest(offline: bool = False, seed: int = DETERMINISTIC_SEED) -> IngestedData:
    """Ingest both datasets using the primary source with a synthetic fallback.

    Args:
        offline: When ``True``, do not attempt network access.
        seed: The deterministic seed used by the synthetic fallback.

    Returns:
        An ``IngestedData`` instance with pricing, skincare, and provenance.
    """
    raw = _load_raw_sephora(offline)
    if raw is not None:
        return IngestedData(
            pricing=parse_sephora_pricing(raw),
            skincare=derive_skincare_compatibility(raw),
            source="primary:sephora",
        )
    pricing, skincare = generate_synthetic(seed)
    return IngestedData(pricing=pricing, skincare=skincare, source="synthetic")
