"""Leak-free stratified splitting and cross-contamination verification."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_FRACTION: float = 0.70
VALIDATION_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15


def _representative_label(frame: pd.DataFrame, target: str, product_col: str) -> pd.Series:
    """Return one stratification label per product.

    Args:
        frame: The feature dataframe.
        target: The target column name.
        product_col: The column that groups rows belonging to one product.

    Returns:
        A series indexed by product identifier holding a representative label.
    """
    grouped = frame.groupby(product_col)[target]
    if pd.api.types.is_numeric_dtype(frame[target]):
        return grouped.first()
    return grouped.agg(lambda values: values.mode().iloc[0])


def _stratify_values(labels: pd.Series) -> pd.Series:
    """Convert representative labels into a stratification-friendly series.

    Args:
        labels: Representative labels indexed by product identifier.

    Returns:
        Either the labels themselves or quantile bins for continuous targets.
    """
    if pd.api.types.is_numeric_dtype(labels):
        return pd.qcut(labels, q=10, duplicates="drop")
    return labels


def stratified_split(
    frame: pd.DataFrame,
    target: str,
    product_col: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into stratified 70/15/15 train/val/test splits by product.

    Grouping by ``product_col`` guarantees zero leakage at both the row and
    product level, so no single product appears in more than one split.

    Args:
        frame: The feature dataframe.
        target: The target column name.
        product_col: The column that groups rows belonging to one product.
        seed: The deterministic random seed.

    Returns:
        A tuple of ``(train, validation, test)`` dataframes.
    """
    labels = _representative_label(frame, target, product_col)
    stratify = _stratify_values(labels)
    strat_df = pd.DataFrame({"product": labels.index, "label": stratify.to_numpy()})

    validation_size = VALIDATION_FRACTION / (TRAIN_FRACTION + VALIDATION_FRACTION)
    try:
        train_df, test_df = train_test_split(
            strat_df,
            test_size=TEST_FRACTION,
            stratify=strat_df["label"],
            random_state=seed,
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=validation_size,
            stratify=train_df["label"],
            random_state=seed,
        )
    except ValueError:
        train_df, test_df = train_test_split(strat_df, test_size=TEST_FRACTION, random_state=seed)
        train_df, val_df = train_test_split(train_df, test_size=validation_size, random_state=seed)

    train_ids = set(train_df["product"])
    val_ids = set(val_df["product"])
    test_ids = set(test_df["product"])

    return (
        frame[frame[product_col].isin(train_ids)].reset_index(drop=True),
        frame[frame[product_col].isin(val_ids)].reset_index(drop=True),
        frame[frame[product_col].isin(test_ids)].reset_index(drop=True),
    )


def verify_no_cross_contamination(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    product_col: str,
) -> None:
    """Assert zero product overlap across the three splits.

    Args:
        train: The training split.
        val: The validation split.
        test: The test split.
        product_col: The column identifying a unique product.

    Raises:
        ValueError: If any product appears in more than one split.
    """
    train_ids = set(train[product_col])
    val_ids = set(val[product_col])
    test_ids = set(test[product_col])
    overlaps: list[str] = []
    if train_ids & val_ids:
        overlaps.append("train/val")
    if train_ids & test_ids:
        overlaps.append("train/test")
    if val_ids & test_ids:
        overlaps.append("val/test")
    if overlaps:
        raise ValueError(f"cross-contamination detected across {', '.join(overlaps)}")
