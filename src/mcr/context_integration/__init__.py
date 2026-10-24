"""Replaceable memory-to-context integration."""

from mcr.context_integration.default import DefaultMemoryContextIntegrator
from mcr.context_integration.errors import (
    CriticalIntegrationError,
    MemoryContextIntegrationError,
    SourceImageUnavailable,
)
from mcr.context_integration.models import (
    MemoryIntegrationRequest,
    MemoryIntegrationResult,
    IntegrationImageRef,
)
from mcr.context_integration.protocol import MemoryContextIntegrator
from mcr.context_integration.query_aware import QueryAwareMemoryContextIntegrator

__all__ = [
    "DefaultMemoryContextIntegrator",
    "MemoryContextIntegrationError",
    "CriticalIntegrationError",
    "SourceImageUnavailable",
    "IntegrationImageRef",
    "QueryAwareMemoryContextIntegrator",
    "MemoryContextIntegrator",
    "MemoryIntegrationRequest",
    "MemoryIntegrationResult",
]
