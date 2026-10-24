"""Configuration for a local OpenAI-compatible vLLM chat server."""

from pydantic import Field

from mcr.memory.common import DomainModel, NonEmptyStr


class VLLMConfig(DomainModel):
    model: NonEmptyStr = "Qwen/Qwen3-VL-8B-Instruct"
    base_url: NonEmptyStr = "http://127.0.0.1:18000/v1"
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_tokens: int = Field(default=8192, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=2)
