"""Qdrant collection lifecycle and compatibility checks."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from mcr.persistence import PersistenceConfigurationError
from mcr.qdrant.config import DistanceMetric, QdrantConfig


_DISTANCES = {
    DistanceMetric.COSINE: models.Distance.COSINE,
    DistanceMetric.DOT: models.Distance.DOT,
    DistanceMetric.EUCLID: models.Distance.EUCLID,
    DistanceMetric.MANHATTAN: models.Distance.MANHATTAN,
}


async def _ensure_payload_indexes(
    client: AsyncQdrantClient, config: QdrantConfig
) -> None:
    for field in (
        "memory_id",
        "item_type",
        "modality",
        "embedding_model",
        "memory.links.target_memory_id",
    ):
        await client.create_payload_index(
            collection_name=config.collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


async def ensure_collection(client: AsyncQdrantClient, config: QdrantConfig) -> None:
    """Create the collection or reject an incompatible existing schema."""

    exists = await client.collection_exists(config.collection_name)
    expected_distance = _DISTANCES[config.distance]
    if not exists:
        if not config.create_if_missing:
            raise PersistenceConfigurationError(
                f"Qdrant collection {config.collection_name!r} does not exist"
            )
        await client.create_collection(
            collection_name=config.collection_name,
            vectors_config=models.VectorParams(
                size=config.vector_size,
                distance=expected_distance,
            ),
        )
        await _ensure_payload_indexes(client, config)
        return

    info = await client.get_collection(config.collection_name)
    vectors = info.config.params.vectors
    if not isinstance(vectors, models.VectorParams):
        raise PersistenceConfigurationError(
            "MCR requires a Qdrant collection with one unnamed dense vector"
        )
    if vectors.size != config.vector_size or vectors.distance != expected_distance:
        raise PersistenceConfigurationError(
            f"Qdrant collection {config.collection_name!r} has vector config "
            f"size={vectors.size}, distance={vectors.distance}; expected "
            f"size={config.vector_size}, distance={expected_distance}"
        )
    await _ensure_payload_indexes(client, config)

