"""Local vLLM multimodal embedding provider."""

from mcr.embedding.providers.vllm.client import VLLMEmbeddingClient
from mcr.embedding.providers.vllm.config import VLLMEmbeddingConfig

__all__ = ["VLLMEmbeddingClient", "VLLMEmbeddingConfig"]
