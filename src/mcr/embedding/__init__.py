"""Provider-neutral multimodal embedding API."""

from mcr.embedding.errors import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingRequestError,
)
from mcr.embedding.normalize import normalize_vector
from mcr.embedding.protocol import MultimodalEmbeddingClient

__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingError",
    "EmbeddingRequestError",
    "MultimodalEmbeddingClient",
    "normalize_vector",
]
