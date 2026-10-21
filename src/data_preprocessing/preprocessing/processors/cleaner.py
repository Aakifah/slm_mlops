"""Exact deduplication and dataframe validation for the prepared datasets."""

from __future__ import annotations

import pandas as pd


def deduplicate_exact(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate records from a feature dataframe.

    Args:
        frame: The feature dataframe.

    Returns:
        The dataframe with exact duplicates removed.
    """
    return frame.drop_duplicates().reset_index(drop=True)
