"""End-to-end unified multimodal memory retriever."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from mcr.embedding import MultimodalEmbeddingClient, normalize_vector
from mcr.memory import MemoryItem
from mcr.qdrant import DistanceMetric, QdrantConfig
from mcr.retrieval.errors import RetrievalConfigurationError, RetrievalDataError
from mcr.retrieval.models import (
    EmbeddedRetrievalQuery,
    RetrievalConfig,
    RetrievalQuery,
    RetrievedMemory,
)
from mcr.retrieval.route_search import search_route
from mcr.retrieval.scoring import score_candidates


_EXPECTED_DISTANCES = {
    DistanceMetric.COSINE: models.Distance.COSINE,
    DistanceMetric.DOT: models.Distance.DOT,
}


class UnifiedMemoryRetriever:
    def __init__(
        self,
        client: AsyncQdrantClient,
        qdrant: QdrantConfig,
        embedder: MultimodalEmbeddingClient,
    ) -> None:
        self._client = client
        self._qdrant = qdrant
        self._embedder = embedder

    def _validate_space(self) -> None:
        if self._qdrant.distance not in {DistanceMetric.COSINE, DistanceMetric.DOT}:
            raise RetrievalConfigurationError(
                "unified retrieval requires cosine or dot-product distance"
            )
        if self._embedder.model_id != self._qdrant.embedding_model:
            raise RetrievalConfigurationError(
                f"embedding model {self._embedder.model_id!r} does not match "
                f"Qdrant model {self._qdrant.embedding_model!r}"
            )
        if self._embedder.dimension != self._qdrant.vector_size:
            raise RetrievalConfigurationError(
                f"embedding dimension {self._embedder.dimension} does not match "
                f"Qdrant vector size {self._qdrant.vector_size}"
            )

    async def search(
        self,
        query: RetrievalQuery,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedMemory]:
        self._validate_space()
        query_vectors = {}
        if query.texts:
            values = await self._embedder.embed_texts(query.texts)
            if len(values) != len(query.texts):
                raise RetrievalConfigurationError(
                    "embedding client returned an unexpected number of text vectors"
                )
            query_vectors["text"] = [
                normalize_vector(value, self._qdrant.vector_size) for value in values
            ]
        if query.images:
            values = await self._embedder.embed_images(query.images)
            if len(values) != len(query.images):
                raise RetrievalConfigurationError(
                    "embedding client returned an unexpected number of image vectors"
                )
            query_vectors["vision"] = [
                normalize_vector(value, self._qdrant.vector_size) for value in values
            ]

        embedded = EmbeddedRetrievalQuery(
            text_vectors=query_vectors.get("text", []),
            vision_vectors=query_vectors.get("vision", []),
            text_vector_weights=query.text_vector_weights,
            vision_vector_weights=query.vision_vector_weights,
            text_modality_weight=query.text_modality_weight,
            vision_modality_weight=query.vision_modality_weight,
            target=query.target,
        )
        return await self.search_vectors(embedded, config)

    async def search_vectors(
        self,
        query: EmbeddedRetrievalQuery,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedMemory]:
        self._validate_space()
        config = config or RetrievalConfig()
        if not await self._client.collection_exists(self._qdrant.collection_name):
            return []
        info = await self._client.get_collection(self._qdrant.collection_name)
        vectors_config = info.config.params.vectors
        if not isinstance(vectors_config, models.VectorParams):
            raise RetrievalConfigurationError("expected one unnamed dense vector")
        if (
            vectors_config.size != self._qdrant.vector_size
            or vectors_config.distance != _EXPECTED_DISTANCES[self._qdrant.distance]
        ):
            raise RetrievalConfigurationError("Qdrant vector configuration is incompatible")

        query_vectors = {}
        if query.text_vectors:
            query_vectors["text"] = [
                normalize_vector(value, self._qdrant.vector_size)
                for value in query.text_vectors
            ]
        if query.vision_vectors:
            query_vectors["vision"] = [
                normalize_vector(value, self._qdrant.vector_size)
                for value in query.vision_vectors
            ]

        routes = []
        for query_modality, vectors in query_vectors.items():
            for target_modality in ("text", "vision"):
                routes.append(
                    await search_route(
                        self._client,
                        self._qdrant,
                        query_modality,
                        target_modality,
                        vectors,
                        query.target,
                        config.vector_top_k,
                        config.cosine_floor,
                    )
                )

        memories = {}
        for route in routes:
            memories.update(route.memories)
        scored = score_candidates(query, routes)
        if config.score_threshold is not None:
            scored = [item for item in scored if item.score >= config.score_threshold]

        results = []
        for rank, item in enumerate(scored[: config.result_top_k], start=1):
            try:
                memory = MemoryItem.model_validate(memories[item.memory_id])
            except Exception as exc:
                raise RetrievalDataError(
                    f"memory {item.memory_id} has an invalid stored payload"
                ) from exc
            results.append(
                RetrievedMemory(
                    memory=memory,
                    score=item.score,
                    rank=rank,
                    item_type=item.item_type,
                    modality_subtotals=item.modality_subtotals,
                    evidence=item.evidence,
                )
            )
        return results
