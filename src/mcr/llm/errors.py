"""Stable errors exposed by LLM provider adapters."""


class LLMError(RuntimeError):
    """Base error for an LLM invocation."""


class LLMConfigurationError(LLMError):
    """The provider is missing required or valid configuration."""


class LLMRequestError(LLMError):
    """The provider rejected or failed an LLM request."""

