"""Qdrant route search and per-memory MaxSim aggregation."""

from __future__ import annotations

from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from mcr.qdrant import QdrantConfig
from mcr.retrieval.errors import RetrievalDataError, RetrievalRequestError
from mcr.retrieval.models import Modality, RouteSearchResult, TargetType


async def search_route(
    client: AsyncQdrantClient,
    qdrant: QdrantConfig,
    query_modality: Modality,
    target_modality: Modality,
    vectors: list[list[float]],
    target: TargetType,
    top_k: int,
    floor: float,
) -> RouteSearchResult:
    conditions = [
        models.FieldCondition(
            key="modality", match=models.MatchValue(value=target_modality)
        ),
        models.FieldCondition(
            key="embedding_model",
            match=models.MatchValue(value=qdrant.embedding_model),
        ),
    ]
    if target != "all":
        conditions.append(
            models.FieldCondition(key="item_type", match=models.MatchValue(value=target))
        )

    scores: list[dict[UUID, float]] = []
    missing_scores: list[float] = []
    item_types = {}
    memories = {}
    try:
        for vector in vectors:
            response = await client.query_points(
                collection_name=qdrant.collection_name,
                query=vector,
                query_filter=models.Filter(must=conditions),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points
            per_memory: dict[UUID, float] = {}
            for point in points:
                payload = point.payload or {}
                try:
                    memory_id = UUID(payload["memory_id"])
                    item_type = payload["item_type"]
                    memory_payload = payload["memory"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise RetrievalDataError("Qdrant point has malformed MCR payload") from exc
                per_memory[memory_id] = max(per_memory.get(memory_id, -1.0), point.score)
                item_types[memory_id] = item_type
                memories[memory_id] = memory_payload
            scores.append(per_memory)
            missing_scores.append(points[-1].score if points else floor)
    except RetrievalDataError:
        raise
    except Exception as exc:
        raise RetrievalRequestError(
            f"Qdrant route {query_modality}->{target_modality} failed"
        ) from exc
    return RouteSearchResult(
        query_modality=query_modality,
        target_modality=target_modality,
        scores=scores,
        missing_scores=missing_scores,
        item_types=item_types,
        memories=memories,
    )
