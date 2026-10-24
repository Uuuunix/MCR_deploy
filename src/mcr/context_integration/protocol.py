"""Provider-neutral memory context integration protocol."""

from __future__ import annotations

from typing import Protocol

from mcr.context_integration.models import (
    MemoryIntegrationRequest,
    MemoryIntegrationResult,
)


class MemoryContextIntegrator(Protocol):
    async def integrate(
        self, request: MemoryIntegrationRequest
    ) -> MemoryIntegrationResult: ...

