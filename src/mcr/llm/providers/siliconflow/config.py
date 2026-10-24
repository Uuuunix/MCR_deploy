"""SiliconFlow provider configuration."""

from __future__ import annotations

import os

from pydantic import Field, SecretStr, model_validator

from mcr.memory.common import DomainModel, NonEmptyStr


class SiliconFlowConfig(DomainModel):
    model: NonEmptyStr
    api_key: SecretStr | None = Field(default=None, repr=False)
    base_url: NonEmptyStr = "https://api.siliconflow.cn/v1"
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    # Extraction responses can contain several memories and multiple retrieval
    # facets per memory.  2048 tokens is too small for ordinary user/assistant
    # pairs and can cut otherwise valid JSON in the middle of a field.
    max_tokens: int = Field(default=8192, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def load_api_key(self) -> "SiliconFlowConfig":
        if self.api_key is None:
            value = os.getenv("SILICONFLOW_API_KEY")
            if value:
                object.__setattr__(self, "api_key", SecretStr(value))
        return self

