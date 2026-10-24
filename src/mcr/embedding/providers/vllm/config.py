"""Configuration for Qwen3-VL-Embedding served by vLLM."""

from pydantic import Field

from mcr.memory.common import DomainModel, NonEmptyStr


class VLLMEmbeddingConfig(DomainModel):
    model: NonEmptyStr = "Qwen/Qwen3-VL-Embedding-2B"
    dimension: int = Field(default=2048, gt=0)
    base_url: NonEmptyStr = "http://127.0.0.1:43227/v1"
    instruction: NonEmptyStr = "Represent the user's input."
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
