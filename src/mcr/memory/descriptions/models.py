"""Core and dynamic retrieval descriptions."""

from __future__ import annotations

import math
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from mcr.memory.common import DomainModel, NonEmptyStr


class RetrievalDescription(DomainModel):
    """One independently embedded description belonging to one memory."""

    id: UUID = Field(default_factory=uuid4)
    facet_type: Literal["core", "dynamic"]
    text: NonEmptyStr
    embedding: list[float] | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
    )

    @field_validator("embedding")
    @classmethod
    def require_finite_embedding(
        cls, value: list[float] | None
    ) -> list[float] | None:
        if value is not None and not all(math.isfinite(item) for item in value):
            raise ValueError("embedding components must be finite")
        return value
