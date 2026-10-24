"""Qdrant implementation of the memory persistence port."""

from __future__ import annotations

from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from mcr.embedding import MultimodalEmbeddingClient
from mcr.memory import MemoryItem, MemoryLink
from mcr.persistence import (
    PersistenceConfigurationError,
    PersistenceError,
    PersistenceWriteError,
)
from mcr.qdrant.config import QdrantConfig
from mcr.qdrant.mapping import memory_to_points
from mcr.qdrant.models import MemoryVector, MemoryVectors
from mcr.qdrant.schema import ensure_collection


class QdrantMemoryRepository:
    """Persist each retrieval description as a vector for the same memory."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        config: QdrantConfig,
        embedder: MultimodalEmbeddingClient | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._embedder = embedder

    async def _visual_vectors(self, memory: MemoryItem) -> dict[UUID, list[float]]:
        image_relations = [
            relation for relation in memory.relations if relation.source_type == "image"
        ]
        if not image_relations:
            return {}
        if self._embedder is None:
            raise PersistenceConfigurationError(
                "an embedding client is required to save image memories"
            )
        if self._embedder.model_id != self._config.embedding_model:
            raise PersistenceConfigurationError(
                f"embedding model {self._embedder.model_id!r} does not match "
                f"Qdrant model {self._config.embedding_model!r}"
            )
        if self._embedder.dimension != self._config.vector_size:
            raise PersistenceConfigurationError(
                f"embedding dimension {self._embedder.dimension} does not match "
                f"Qdrant vector size {self._config.vector_size}"
            )
        missing = [relation.id for relation in image_relations if relation.image_source is None]
        if missing:
            raise PersistenceConfigurationError(
                f"image relations require accessible image_source values: {missing}"
            )
        vectors = await self._embedder.embed_images(
            [relation.image_source for relation in image_relations if relation.image_source]
        )
        if len(vectors) != len(image_relations):
            raise PersistenceConfigurationError(
                "embedding client returned an unexpected number of image vectors"
            )
        return {
            relation.id: vector for relation, vector in zip(image_relations, vectors, strict=True)
        }

    async def save(
        self,
        memory: MemoryItem,
        visual_vectors: dict[UUID, list[float]] | None = None,
    ) -> None:
        try:
            if visual_vectors is None:
                visual_vectors = await self._visual_vectors(memory)
            points = memory_to_points(
                memory,
                self._config.vector_size,
                self._config.embedding_model,
                visual_vectors,
            )
            await ensure_collection(self._client, self._config)
            await self._client.delete(
                collection_name=self._config.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="memory_id",
                                match=models.MatchValue(value=str(memory.id)),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            await self._client.upsert(
                collection_name=self._config.collection_name,
                points=points,
                wait=True,
            )
        except PersistenceError:
            raise
        except Exception as exc:
            raise PersistenceWriteError(
                f"failed to save memory {memory.id} to Qdrant"
            ) from exc

    async def get_memory(self, memory_id: UUID | str) -> MemoryItem | None:
        if not await self._client.collection_exists(self._config.collection_name):
            return None
        records, _ = await self._client.scroll(
            collection_name=self._config.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="memory_id", match=models.MatchValue(value=str(memory_id))
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        try:
            return MemoryItem.model_validate(payload["memory"])
        except Exception as exc:
            raise PersistenceConfigurationError(
                f"memory {memory_id} has malformed payload"
            ) from exc

    async def list_memories(self) -> list[MemoryItem]:
        """Return each logical memory once, regardless of its vector-point count."""
        if not await self._client.collection_exists(self._config.collection_name):
            return []
        memories: dict[UUID, MemoryItem] = {}
        offset = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._config.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                try:
                    memory = MemoryItem.model_validate(payload["memory"])
                except Exception as exc:
                    raise PersistenceConfigurationError(
                        f"Qdrant point {record.id} has malformed memory payload"
                    ) from exc
                memories.setdefault(memory.id, memory)
            if offset is None:
                break
        return sorted(
            memories.values(),
            key=lambda memory: (memory.meta.created_at, str(memory.id)),
            reverse=True,
        )

    async def update_links(
        self, memory_id: UUID | str, links: list[MemoryLink]
    ) -> None:
        memory = await self.get_memory(memory_id)
        if memory is None:
            raise PersistenceConfigurationError(f"memory {memory_id} does not exist")
        memory.links = links
        await self._client.set_payload(
            collection_name=self._config.collection_name,
            payload={"memory": memory.model_dump(mode="json")},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="memory_id", match=models.MatchValue(value=str(memory_id))
                    )
                ]
            ),
            wait=True,
        )

    async def delete_memory(self, memory_id: UUID | str) -> None:
        await self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="memory_id", match=models.MatchValue(value=str(memory_id))
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def get_vectors(self, memory: MemoryItem | UUID | str) -> MemoryVectors:
        memory_id = memory.id if isinstance(memory, MemoryItem) else UUID(str(memory))
        records = []
        offset = None
        while True:
            page, offset = await self._client.scroll(
                collection_name=self._config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="memory_id",
                            match=models.MatchValue(value=str(memory_id)),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            records.extend(page)
            if offset is None:
                break

        grouped = MemoryVectors(memory_id=memory_id)
        for record in records:
            payload = record.payload or {}
            vector = record.vector
            if not isinstance(vector, list):
                raise PersistenceConfigurationError("expected one unnamed dense vector")
            item = MemoryVector(
                point_id=UUID(str(record.id)),
                memory_id=memory_id,
                modality=payload["modality"],
                local_id=UUID(payload["local_id"]),
                vector=vector,
                facet_type=payload.get("facet_type"),
            )
            getattr(grouped, item.modality).append(item)
            grouped.embedding_model = payload.get("embedding_model")
        return grouped

