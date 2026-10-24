"""Stable embedding-layer exceptions."""


class EmbeddingError(RuntimeError):
    """Base class for embedding failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Embedding configuration or input is invalid."""


class EmbeddingRequestError(EmbeddingError):
    """The embedding provider request failed."""
