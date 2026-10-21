"""Build the FAISS schema vector index from persisted RAG chunks.

Reads ``src/artifacts/rag_chunks.json`` (validated by ``RAGChunkSchema``),
embeds every chunk with the deterministic local embedder, and writes both the
FAISS index and its metadata into ``src/artifacts/faiss_store/``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from src.backend.schemas.api import RAGChunkSchema

from src.data_preprocessing.preprocessing.paths import FAISS_STORE_DIR, RAG_CHUNKS_PATH
from src.model_customization.rag.embedding import EMBEDDING_DIM, embed_texts

_INDEX_PATH: Path = FAISS_STORE_DIR / "index.faiss"
_METADATA_PATH: Path = FAISS_STORE_DIR / "index.pkl"


def _load_chunks(path: Path) -> RAGChunkSchema:
    """Load and validate the persisted RAG chunk artifact.

    Args:
        path: The source JSON path.

    Returns:
        The validated chunk schema.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RAGChunkSchema.model_validate(payload)


def build_index() -> int:
    """Build and persist the FAISS schema index.

    Returns:
        The number of chunks embedded into the index.
    """
    chunks = _load_chunks(RAG_CHUNKS_PATH)
    vectors = embed_texts([chunk.text for chunk in chunks.chunks])
    vectors = vectors.astype("float32")
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"embedding dimension {vectors.shape[1]} != {EMBEDDING_DIM}")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.ascontiguousarray(vectors))

    FAISS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_INDEX_PATH))

    metadata = {
        "dimension": EMBEDDING_DIM,
        "chunk_ids": [chunk.id for chunk in chunks.chunks],
    }
    _METADATA_PATH.write_bytes(pickle.dumps(metadata))

    return len(chunks.chunks)


if __name__ == "__main__":
    raise SystemExit(0 if build_index() > 0 else 1)
