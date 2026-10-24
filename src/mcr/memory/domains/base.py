"""Shared fields for every schema-v2 d_j description facet."""

from __future__ import annotations

import math
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from mcr.memory.common import DomainModel, NonEmptyStr


class DescriptionBase(DomainModel):
    """Fields present on every independently embedded d_j."""

    id: UUID = Field(default_factory=uuid4)
    text: NonEmptyStr
    embedding: list[float] | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
    )
    @field_validator("embedding")
    @classmethod
    def require_finite_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not all(math.isfinite(component) for component in value):
            raise ValueError("embedding components must be finite")
        return value

