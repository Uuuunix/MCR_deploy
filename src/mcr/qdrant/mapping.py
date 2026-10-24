"""Pure mapping from a MemoryItem to Qdrant description points."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import models

from mcr.embedding import EmbeddingConfigurationError, normalize_vector
from mcr.memory import MemoryItem
from mcr.memory import RetrievalDescription
from mcr.persistence import PersistenceConfigurationError


def description_point_id(memory_id: UUID, description_id: UUID) -> UUID:
    """Create a stable collection-wide point ID for one d_j."""

    return uuid5(NAMESPACE_URL, f"mcr:{memory_id}:text:{description_id}")


def visual_point_id(memory_id: UUID, relation_id: UUID) -> UUID:
    """Create a stable point ID for one image relation vector."""

    return uuid5(NAMESPACE_URL, f"mcr:{memory_id}:vision:{relation_id}")


def iter_descriptions(
    memory: MemoryItem,
) -> Iterator[tuple[int, RetrievalDescription]]:
    yield from enumerate(memory.descriptions)


def _descriptions_without_embeddings(memory: MemoryItem) -> list[dict[str, Any]]:
    descriptions = deepcopy(memory.model_dump(mode="json")["descriptions"])
    for item in descriptions:
        item.pop("embedding", None)
    return descriptions


def memory_to_points(
    memory: MemoryItem,
    vector_size: int,
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-8B",
    visual_vectors: dict[UUID, list[float]] | None = None,
) -> list[models.PointStruct]:
    """Map every d_j in a memory to a self-contained Qdrant point."""

    memory_dump = memory.model_dump(mode="json")
    canonical_memory = {
        "schema_version": memory_dump["schema_version"],
        "id": str(memory.id),
        "x": memory_dump["x"],
        "meta": memory_dump["meta"],
        "descriptions": _descriptions_without_embeddings(memory),
        "relations": memory_dump["relations"],
        "links": memory_dump["links"],
    }
    item_type = (
        "image" if any(relation.source_type == "image" for relation in memory.relations) else "text"
    )
    canonical_payload = {
        "memory_id": str(memory.id),
        "item_type": item_type,
        "embedding_model": embedding_model,
        "memory": canonical_memory,
        # Kept for backward-compatible payload consumers.
        **{
            key: canonical_memory[key]
            for key in (
                "schema_version",
                "x",
                "meta",
                "descriptions",
                "relations",
                "links",
            )
        },
    }

    points: list[models.PointStruct] = []
    for facet_index, description in iter_descriptions(memory):
        if description.embedding is None:
            raise PersistenceConfigurationError(
                f"description {description.id} has no embedding"
            )
        try:
            vector = normalize_vector(description.embedding, vector_size)
        except EmbeddingConfigurationError as exc:
            raise PersistenceConfigurationError(
                f"invalid embedding for description {description.id}: {exc}"
            ) from exc

        description_data = description.model_dump(mode="json")
        description_data.pop("embedding")
        payload = {
            **canonical_payload,
            "modality": "text",
            "local_id": str(description.id),
            "facet_index": facet_index,
            "facet_type": description.facet_type,
            "description_id": str(description.id),
            "description_text": description.text,
            "description": description_data,
        }
        points.append(
            models.PointStruct(
                id=str(description_point_id(memory.id, description.id)),
                vector=vector,
                payload=payload,
            )
        )

    for relation in memory.relations:
        if relation.source_type != "image":
            continue
        vector_value = (visual_vectors or {}).get(relation.id)
        if vector_value is None:
            raise PersistenceConfigurationError(
                f"image relation {relation.id} has no visual embedding"
            )
        try:
            vector = normalize_vector(vector_value, vector_size)
        except EmbeddingConfigurationError as exc:
            raise PersistenceConfigurationError(
                f"invalid embedding for image relation {relation.id}: {exc}"
            ) from exc
        payload = {
            **canonical_payload,
            "modality": "vision",
            "local_id": str(relation.id),
            "facet_type": None,
            "relation_id": str(relation.id),
        }
        points.append(
            models.PointStruct(
                id=str(visual_point_id(memory.id, relation.id)),
                vector=vector,
                payload=payload,
            )
        )
    return points

