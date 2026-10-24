"""SiliconFlow embedding configuration."""

from __future__ import annotations

import os

from pydantic import Field, SecretStr, model_validator

from mcr.memory.common import DomainModel, NonEmptyStr


class SiliconFlowEmbeddingConfig(DomainModel):
    model: NonEmptyStr = "Qwen/Qwen3-VL-Embedding-8B"
    dimension: int = Field(gt=0)
    api_key: SecretStr | None = Field(default=None, repr=False)
    base_url: NonEmptyStr = "https://api.siliconflow.cn/v1"
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def load_api_key(self) -> "SiliconFlowEmbeddingConfig":
        if self.api_key is None:
            value = os.getenv("SILICONFLOW_API_KEY")
            if value:
                object.__setattr__(self, "api_key", SecretStr(value))
        return self
