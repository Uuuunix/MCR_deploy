"""Pure implementation of the unified multimodal scoring equation."""

from __future__ import annotations

from uuid import UUID

from mcr.retrieval.models import (
    Modality,
    EmbeddedRetrievalQuery,
    RetrievalQuery,
    RouteEvidence,
    RouteSearchResult,
    ScoredCandidate,
)


def score_candidates(
    query: RetrievalQuery | EmbeddedRetrievalQuery,
    routes: list[RouteSearchResult],
) -> list[ScoredCandidate]:
    route_map = {(route.query_modality, route.target_modality): route for route in routes}
    candidates: set[UUID] = set()
    item_types = {}
    for route in routes:
        for vector_scores in route.scores:
            candidates.update(vector_scores)
        item_types.update(route.item_types)

    modality_weights = query.modality_weights()
    results = []
    for memory_id in candidates:
        item_type = item_types[memory_id]
        target_modalities: tuple[Modality, ...] = (
            ("text", "vision") if item_type == "image" else ("text",)
        )
        subtotals: dict[Modality, float] = {}
        evidence: list[RouteEvidence] = []
        for query_modality in modality_weights:
            vector_weights = query.vector_weights(query_modality)
            subtotal = 0.0
            for index, vector_weight in enumerate(vector_weights):
                choices = []
                for target_modality in target_modalities:
                    route = route_map[(query_modality, target_modality)]
                    observed = memory_id in route.scores[index]
                    value = route.scores[index].get(
                        memory_id, route.missing_scores[index]
                    )
                    choices.append((value, target_modality, not observed))
                best_score, best_target, backfilled = max(choices, key=lambda item: item[0])
                subtotal += vector_weight * best_score
                evidence.append(
                    RouteEvidence(
                        query_modality=query_modality,
                        query_vector_index=index,
                        target_modality=best_target,
                        score=best_score,
                        backfilled=backfilled,
                    )
                )
            subtotals[query_modality] = subtotal
        total = sum(modality_weights[key] * value for key, value in subtotals.items())
        results.append(
            ScoredCandidate(
                memory_id=memory_id,
                score=total,
                item_type=item_type,
                modality_subtotals=subtotals,
                evidence=evidence,
            )
        )
    return sorted(results, key=lambda result: (-result.score, str(result.memory_id)))
