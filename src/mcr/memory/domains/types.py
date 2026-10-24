"""One D schema shared by the six queryable memory domains."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from mcr.memory.common import DomainModel
from mcr.memory.domains.facets import DescriptionFacet

DomainName = Literal[
    "biography",
    "experiences",
    "preferences",
    "social_circle",
    "work",
    "psychometrics",
]


class MemoryDomain(DomainModel):
    domain: DomainName
    items: list[DescriptionFacet] = Field(min_length=1)

    @model_validator(mode="after")
    def require_core_facet(self) -> "MemoryDomain":
        if not any(item.facet_type == "core" for item in self.items):
            raise ValueError("each domain D requires at least one core facet")
        return self

