"""Local vLLM chat provider."""

from mcr.llm.providers.vllm.client import VLLMClient
from mcr.llm.providers.vllm.config import VLLMConfig

__all__ = ["VLLMClient", "VLLMConfig"]
