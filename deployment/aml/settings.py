"""Validated configuration for the academic multimodal deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path
    memory_system_key: str | None
    openai_api_key: str | None
    dashscope_api_key: str | None
    openai_base_url: str = "https://api.openai.com/v1"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_dimension: int = 1024
    recent_window_size: int = 8
    api_prefix: str = ""
    llm_model: str = "gpt-4o-mini"

    @classmethod
    def from_environment(cls) -> "Settings":
        dimension = int(os.getenv("AML_EMBEDDING_DIMENSION", "1024"))
        if dimension not in {64, 128, 256, 512, 768, 1024, 1536, 2048}:
            raise ValueError("AML_EMBEDDING_DIMENSION is unsupported by text-embedding-v4")
        recent_window_size = int(os.getenv("AML_RECENT_WINDOW_SIZE", "8"))
        if recent_window_size < 0:
            raise ValueError("AML_RECENT_WINDOW_SIZE must not be negative")
        api_prefix = os.getenv("AML_API_PREFIX", "").rstrip("/")
        if api_prefix and (not api_prefix.startswith("/") or "//" in api_prefix):
            raise ValueError("AML_API_PREFIX must be a URL path prefix")
        llm_model = os.getenv("AML_LLM_MODEL", "gpt-4o-mini").strip()
        if not llm_model:
            raise ValueError("AML_LLM_MODEL must not be empty")
        return cls(
            data_root=Path(os.getenv("AML_DATA_ROOT", ".runtime/aml")).expanduser().resolve(),
            memory_system_key=os.getenv("AML_MEMORY_SYSTEM_KEY") or None,
            openai_api_key=os.getenv("AML_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or None,
            openai_base_url=(
                os.getenv("AML_LLM_BASE_URL")
                or os.getenv("AML_OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            dashscope_base_url=os.getenv(
                "AML_DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            embedding_dimension=dimension,
            recent_window_size=recent_window_size,
            api_prefix=api_prefix,
            llm_model=llm_model,
        )

    def require_providers(self) -> None:
        missing = []
        if not self.openai_api_key:
            missing.append("AML_LLM_API_KEY (or OPENAI_API_KEY)")
        if not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY")
        if missing:
            raise RuntimeError("model API credentials missing: " + ", ".join(missing))
