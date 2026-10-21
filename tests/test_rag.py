"""FAISS embedding, index build, and scaler serialization tests."""

from __future__ import annotations

from pathlib import Path

import faiss
import pytest

from src.backend.schemas.api import RegressionScalerSchema, ScalerBounds
from src.data_preprocessing.preprocessing.processors.scaler import (
    load_scaler,
    write_scaler,
)
from src.model_customization.rag.embedding import EMBEDDING_DIM, embed_texts


def test_embed_texts_shape_and_determinism() -> None:
    """Embeddings must be fixed-dimension float32 and deterministic."""
    texts = ["niacinamide serum", "sunscreen spf"]
    vectors = embed_texts(texts)
    assert vectors.shape == (2, EMBEDDING_DIM)
    assert vectors.dtype.name == "float32"
    again = embed_texts(texts)
    assert (vectors == again).all()


def test_scaler_roundtrip(tmp_path: Path) -> None:
    """A scaler must serialize and reload losslessly."""
    schema = RegressionScalerSchema(
        dataset="sephora_pricing",
        features={"price_usd": ScalerBounds(min=3.0, max=395.0)},
    )
    path = tmp_path / "scaler.json"
    write_scaler(schema, path)
    restored = load_scaler(path)
    assert restored == schema


def test_build_index_persists_faiss_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The index builder must embed all chunks and persist the FAISS index."""
    import src.model_customization.rag.index_faiss as index_faiss

    from src.model_customization.rag.chunks import build_rag_chunks

    chunks = build_rag_chunks()
    chunks_path = tmp_path / "rag_chunks.json"
    chunks_path.write_text(chunks.model_dump_json(), encoding="utf-8")

    store = tmp_path / "faiss_store"
    monkeypatch.setattr(index_faiss, "RAG_CHUNKS_PATH", chunks_path)
    monkeypatch.setattr(index_faiss, "FAISS_STORE_DIR", store)
    monkeypatch.setattr(index_faiss, "_INDEX_PATH", store / "index.faiss")
    monkeypatch.setattr(index_faiss, "_METADATA_PATH", store / "index.pkl")

    count = index_faiss.build_index()

    assert count == len(chunks.chunks)
    assert (store / "index.faiss").exists()
    index = faiss.read_index(str(store / "index.faiss"))
    assert index.ntotal == count
