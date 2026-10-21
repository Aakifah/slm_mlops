"""Deterministic local embedding for the offline schema FAISS index.

A character/word hashing vectorizer produces fixed-dimension L2-normalized
vectors so the schema index can be built without any external embedding model.
Runtime semantic retrieval replaces this with ``nomic-embed-text`` in a later
milestone while reusing the same FAISS layout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import HashingVectorizer

EMBEDDING_DIM: int = 384


def _vectorizer() -> HashingVectorizer:
    """Return a freshly constructed deterministic hashing vectorizer.

    Returns:
        A ``HashingVectorizer`` with fixed dimensions and disabled sign
        alternation for cross-run determinism.
    """
    return HashingVectorizer(
        n_features=EMBEDDING_DIM,
        norm="l2",
        alternate_sign=False,
    )


def embed_texts(texts: list[str]) -> NDArray[np.float32]:
    """Embed a list of texts into deterministic float32 vectors.

    Args:
        texts: The texts to embed.

    Returns:
        A float32 array of shape ``(len(texts), EMBEDDING_DIM)``.
    """
    matrix = _vectorizer().fit_transform(texts).toarray()
    return np.asarray(matrix, dtype=np.float32)
