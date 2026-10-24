"""Configuration for the Qdrant persistence adapter."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from mcr.memory.common import DomainModel, NonEmptyStr


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"
    MANHATTAN = "manhattan"


class QdrantConfig(DomainModel):
    collection_name: NonEmptyStr = "mcr_memories"
    vector_size: int = Field(gt=0)
    embedding_model: NonEmptyStr = "Qwen/Qwen3-VL-Embedding-8B"
    distance: DistanceMetric = DistanceMetric.COSINE
    create_if_missing: bool = True

