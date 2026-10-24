"""Qdrant-specific vector trace models."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from mcr.memory.common import DomainModel, NonEmptyStr


class MemoryVector(DomainModel):
    point_id: UUID
    memory_id: UUID
    modality: Literal["text", "vision"]
    local_id: UUID
    vector: list[float] = Field(min_length=1)
    facet_type: Literal["core", "dynamic"] | None = None


class MemoryVectors(DomainModel):
    memory_id: UUID
    embedding_model: NonEmptyStr | None = None
    text: list[MemoryVector] = Field(default_factory=list)
    vision: list[MemoryVector] = Field(default_factory=list)
