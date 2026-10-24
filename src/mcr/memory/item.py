"""The aggregate m_i = {x, meta, D, r}."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from mcr.memory.common import DomainModel
from mcr.memory.content import MemoryContent
from mcr.memory.descriptions import RetrievalDescription
from mcr.memory.metadata import MemoryMetadata
from mcr.memory.links import MemoryLink
from mcr.memory.relations import SourceRelation


class MemoryItem(DomainModel):
    """Canonical structured memory persisted by MCR."""

    schema_version: Literal[2]
    id: UUID = Field(default_factory=uuid4)
    x: MemoryContent
    meta: MemoryMetadata
    descriptions: list[RetrievalDescription] = Field(min_length=1)
    relations: list[SourceRelation] = Field(min_length=1)
    links: list[MemoryLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_nested_values(self) -> "MemoryItem":
        description_ids = [item.id for item in self.descriptions]
        if len(description_ids) != len(set(description_ids)):
            raise ValueError("description IDs must be unique within a memory")
        if sum(item.facet_type == "core" for item in self.descriptions) != 1:
            raise ValueError("a memory requires exactly one core description")

        relation_ids = [relation.id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation IDs must be unique within a memory")

        link_targets = [link.target_memory_id for link in self.links]
        if self.id in link_targets:
            raise ValueError("a memory cannot link to itself")
        if len(link_targets) != len(set(link_targets)):
            raise ValueError("memory link targets must be unique")
        return self

