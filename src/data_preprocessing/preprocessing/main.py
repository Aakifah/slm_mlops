"""Milestone 1 data preparation pipeline entry point.

Executes deterministic ingestion, exact deduplication, a leak-free 70/15/15
split, cross-contamination verification, min-max scaler fitting, and writes all
read-only Parquet and JSON artifacts into ``src/artifacts/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from src.model_customization.rag.chunks import build_rag_chunks

from .parsers.source import ingest
from .paths import (
    ARTIFACTS_DIR,
    DETERMINISTIC_SEED,
    FAISS_STORE_DIR,
    RAG_CHUNKS_PATH,
    REGRESSION_SCALER_PATH,
)
from .processors.cleaner import deduplicate_exact
from .processors.scaler import fit_scaler, write_scaler
from .processors.splitter import stratified_split, verify_no_cross_contamination
from .schemas import (
    PRICING_NUMERIC_COLUMNS,
    PRICING_TARGET,
    SKINCARE_TARGET,
)

_PRODUCT_COLUMN: str = "product_id"


def _is_offline() -> bool:
    """Return whether the pipeline should avoid network access.

    Returns:
        ``True`` when the ``SLM_MLOPS_OFFLINE`` environment variable is ``"1"``.
    """
    return os.environ.get("SLM_MLOPS_OFFLINE", "0") == "1"


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to a Parquet artifact without the pandas index.

    Args:
        frame: The dataframe to write.
        path: The destination path.
    """
    frame.to_parquet(path, index=False)


def main() -> int:
    """Run the full data preparation pipeline.

    Returns:
        The process exit code.
    """
    offline = _is_offline()
    data = ingest(offline=offline, seed=DETERMINISTIC_SEED)

    pricing = deduplicate_exact(data.pricing)
    skincare = deduplicate_exact(data.skincare)

    pricing_train, pricing_val, pricing_test = stratified_split(
        pricing, PRICING_TARGET, _PRODUCT_COLUMN, DETERMINISTIC_SEED
    )
    verify_no_cross_contamination(pricing_train, pricing_val, pricing_test, _PRODUCT_COLUMN)

    skincare_train, skincare_val, skincare_test = stratified_split(
        skincare, SKINCARE_TARGET, _PRODUCT_COLUMN, DETERMINISTIC_SEED
    )
    verify_no_cross_contamination(skincare_train, skincare_val, skincare_test, _PRODUCT_COLUMN)

    scaler = fit_scaler(pricing_train, PRICING_NUMERIC_COLUMNS, "sephora_pricing")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_STORE_DIR.mkdir(parents=True, exist_ok=True)

    _write_parquet(skincare, ARTIFACTS_DIR / "skincare_compatibility_clean.parquet")
    _write_parquet(skincare_train, ARTIFACTS_DIR / "skincare_train.parquet")
    _write_parquet(skincare_val, ARTIFACTS_DIR / "skincare_val.parquet")
    _write_parquet(skincare_test, ARTIFACTS_DIR / "skincare_test.parquet")

    _write_parquet(pricing, ARTIFACTS_DIR / "sephora_pricing_normalized.parquet")
    _write_parquet(pricing_train, ARTIFACTS_DIR / "pricing_train.parquet")
    _write_parquet(pricing_val, ARTIFACTS_DIR / "pricing_val.parquet")
    _write_parquet(pricing_test, ARTIFACTS_DIR / "pricing_test.parquet")

    write_scaler(scaler, REGRESSION_SCALER_PATH)

    chunks = build_rag_chunks()
    RAG_CHUNKS_PATH.write_text(chunks.model_dump_json(indent=2), encoding="utf-8")

    print(f"ingestion source       : {data.source}")
    print(f"pricing rows           : {len(pricing)}")
    print(f"pricing split          : {len(pricing_train)}/{len(pricing_val)}/{len(pricing_test)}")
    print(f"skincare rows          : {len(skincare)}")
    print(
        f"skincare split         : {len(skincare_train)}/{len(skincare_val)}/{len(skincare_test)}"
    )
    print(f"skincare tier balance  : {dict(skincare[SKINCARE_TARGET].value_counts())}")
    print(f"scaler features        : {sorted(scaler.features)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
