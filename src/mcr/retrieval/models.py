"""Public query, route, and result models."""

from __future__ import annotations

import math
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from mcr.memory import ImageSource, MemoryItem
from mcr.memory.common import DomainModel, NonEmptyStr

Modality = Literal["text", "vision"]
TargetType = Literal["all", "text", "image"]


class RetrievalQuery(DomainModel):
    texts: list[NonEmptyStr] = Field(default_factory=list)
    images: list[ImageSource] = Field(default_factory=list)
    text_vector_weights: list[float] | None = None
    vision_vector_weights: list[float] | None = None
    text_modality_weight: float | None = Field(default=None, ge=0)
    vision_modality_weight: float | None = Field(default=None, ge=0)
    target: TargetType = "all"

    @model_validator(mode="after")
    def validate_query(self) -> "RetrievalQuery":
        if not self.texts and not self.images:
            raise ValueError("retrieval query requires text or image input")
        self._validate_weights("text", self.text_vector_weights, len(self.texts))
        self._validate_weights("vision", self.vision_vector_weights, len(self.images))
        active = []
        if self.texts:
            active.append(self.text_modality_weight)
        if self.images:
            active.append(self.vision_modality_weight)
        if any(value is not None for value in active):
            if any(value is None for value in active):
                raise ValueError("weights are required for every active query modality")
            if not math.isclose(sum(value or 0 for value in active), 1.0, abs_tol=1e-6):
                raise ValueError("active query modality weights must sum to 1")
        return self

    @staticmethod
    def _validate_weights(name: str, weights: list[float] | None, count: int) -> None:
        if weights is None:
            return
        if len(weights) != count:
            raise ValueError(f"{name} vector weights must match input count")
        if not weights or any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError(f"{name} vector weights must be finite and non-negative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError(f"{name} vector weights must sum to 1")

    def vector_weights(self, modality: Modality) -> list[float]:
        values = self.texts if modality == "text" else self.images
        configured = (
            self.text_vector_weights if modality == "text" else self.vision_vector_weights
        )
        return configured or [1.0 / len(values)] * len(values)

    def modality_weights(self) -> dict[Modality, float]:
        if self.texts and self.images:
            return {
                "text": self.text_modality_weight if self.text_modality_weight is not None else 0.5,
                "vision": self.vision_modality_weight if self.vision_modality_weight is not None else 0.5,
            }
        if self.texts:
            return {"text": 1.0}
        return {"vision": 1.0}


class RetrievalConfig(DomainModel):
    vector_top_k: int = Field(default=1000, gt=0)
    result_top_k: int = Field(default=10, gt=0)
    score_threshold: float | None = None
    cosine_floor: float = Field(default=-1.0, ge=-1.0, le=1.0)


class EmbeddedRetrievalQuery(DomainModel):
    """A query whose vectors were already generated for persistence."""

    text_vectors: list[list[float]] = Field(default_factory=list)
    vision_vectors: list[list[float]] = Field(default_factory=list)
    text_vector_weights: list[float] | None = None
    vision_vector_weights: list[float] | None = None
    text_modality_weight: float | None = Field(default=None, ge=0)
    vision_modality_weight: float | None = Field(default=None, ge=0)
    target: TargetType = "all"

    @model_validator(mode="after")
    def validate_query(self) -> "EmbeddedRetrievalQuery":
        if not self.text_vectors and not self.vision_vectors:
            raise ValueError("embedded query requires text or vision vectors")
        RetrievalQuery._validate_weights(
            "text", self.text_vector_weights, len(self.text_vectors)
        )
        RetrievalQuery._validate_weights(
            "vision", self.vision_vector_weights, len(self.vision_vectors)
        )
        active = []
        if self.text_vectors:
            active.append(self.text_modality_weight)
        if self.vision_vectors:
            active.append(self.vision_modality_weight)
        if any(value is not None for value in active):
            if any(value is None for value in active):
                raise ValueError("weights are required for every active query modality")
            if not math.isclose(sum(value or 0 for value in active), 1.0, abs_tol=1e-6):
                raise ValueError("active query modality weights must sum to 1")
        return self

    def vector_weights(self, modality: Modality) -> list[float]:
        values = self.text_vectors if modality == "text" else self.vision_vectors
        configured = (
            self.text_vector_weights if modality == "text" else self.vision_vector_weights
        )
        return configured or [1.0 / len(values)] * len(values)

    def modality_weights(self) -> dict[Modality, float]:
        if self.text_vectors and self.vision_vectors:
            return {
                "text": self.text_modality_weight if self.text_modality_weight is not None else 0.5,
                "vision": self.vision_modality_weight if self.vision_modality_weight is not None else 0.5,
            }
        if self.text_vectors:
            return {"text": 1.0}
        return {"vision": 1.0}


class RouteSearchResult(DomainModel):
    query_modality: Modality
    target_modality: Modality
    scores: list[dict[UUID, float]]
    missing_scores: list[float]
    item_types: dict[UUID, Literal["text", "image"]] = Field(default_factory=dict)
    memories: dict[UUID, dict] = Field(default_factory=dict)


class RouteEvidence(DomainModel):
    query_modality: Modality
    query_vector_index: int = Field(ge=0)
    target_modality: Modality
    score: float
    backfilled: bool


class RetrievedMemory(DomainModel):
    memory: MemoryItem
    score: float
    rank: int = Field(gt=0)
    item_type: Literal["text", "image"]
    modality_subtotals: dict[Modality, float]
    evidence: list[RouteEvidence]


class ScoredCandidate(DomainModel):
    memory_id: UUID
    score: float
    item_type: Literal["text", "image"]
    modality_subtotals: dict[Modality, float]
    evidence: list[RouteEvidence]
