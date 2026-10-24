"""Dense-vector validation and normalization."""

from __future__ import annotations

import math
from collections.abc import Iterable

from mcr.embedding.errors import EmbeddingConfigurationError


def normalize_vector(vector: Iterable[float], dimension: int) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != dimension:
        raise EmbeddingConfigurationError(
            f"embedding has dimension {len(values)}; expected {dimension}"
        )
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingConfigurationError("embedding components must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise EmbeddingConfigurationError("embedding must not be a zero vector")
    return [value / norm for value in values]
