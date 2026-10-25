"""Filesystem paths shared across the data preparation pipeline.

These constants centralize the locations of the read-only artifact directory,
the optional local raw-data staging directory, and the upstream data source so
that the pipeline and the FAISS index builder resolve paths identically.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

ARTIFACTS_DIR: Path = PROJECT_ROOT / "src" / "artifacts"
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
FAISS_STORE_DIR: Path = ARTIFACTS_DIR / "faiss_store"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
EVALUATION_RESULTS_DIR: Path = ARTIFACTS_DIR / "evaluation_results"

REGRESSION_SCALER_PATH: Path = ARTIFACTS_DIR / "regression_scaler.json"
RAG_CHUNKS_PATH: Path = ARTIFACTS_DIR / "rag_chunks.json"

SEPHORA_REPO_ID: str = "MayaKitzis/sephora_products"
SEPHORA_FILENAME: str = "product_info.csv"
SEPHORA_LOCAL_CSV: Path = RAW_DATA_DIR / "product_info.csv"

DETERMINISTIC_SEED: int = 42
