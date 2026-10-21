"""Artifact contract schemas for data preparation and retrieval.

These Pydantic v2 models validate every JSON artifact loaded from disk before
it is consumed by the application or by downstream pipeline stages. They are
the deterministic boundary between the immutable ``src/artifacts/`` directory
and the rest of the system.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

_DATASET_NAMES: frozenset[str] = frozenset({"skincare_compatibility", "sephora_pricing"})


class ScalerBounds(BaseModel):
    """Min and max normalization bounds for a single numerical feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: float
    max: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ScalerBounds:
        """Reject degenerate bounds where ``min`` is not strictly below ``max``.

        Returns:
            The validated bounds instance.

        Raises:
            ValueError: If ``min >= max``.
        """
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be strictly less than max ({self.max})")
        return self


class RegressionScalerSchema(BaseModel):
    """Contract for ``src/artifacts/regression_scaler.json``.

    Persists the per-feature min/max bounds fitted strictly on the 70% training
    split so that normalized model predictions can be deterministically
    denormalized back into real-world units.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    features: dict[str, ScalerBounds]

    @model_validator(mode="after")
    def validate_dataset(self) -> RegressionScalerSchema:
        """Ensure the scaler targets a known regression dataset.

        Returns:
            The validated scaler schema instance.

        Raises:
            ValueError: If ``dataset`` is not a supported dataset name.
        """
        if self.dataset not in _DATASET_NAMES:
            raise ValueError(f"unknown dataset {self.dataset!r}")
        return self


class RAGChunk(BaseModel):
    """A single schema-metadata chunk embedded into the FAISS index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    dataset: str
    text: str
    columns: list[str]
    aliases: list[str]


class RAGChunkSchema(BaseModel):
    """Contract for ``src/artifacts/rag_chunks.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunks: list[RAGChunk]

    @model_validator(mode="after")
    def validate_chunks(self) -> RAGChunkSchema:
        """Ensure every chunk references a supported dataset.

        Returns:
            The validated chunk schema instance.

        Raises:
            ValueError: If a chunk references an unsupported dataset.
        """
        unknown = {chunk.dataset for chunk in self.chunks} - _DATASET_NAMES
        if unknown:
            raise ValueError(f"unknown datasets {sorted(unknown)}")
        return self
