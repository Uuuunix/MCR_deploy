"""Backend-independent write contract."""

from __future__ import annotations

from typing import Protocol

from mcr.memory import MemoryItem


class MemoryRepository(Protocol):
    """Persistence port implemented by concrete storage adapters."""

    async def save(self, memory: MemoryItem) -> None:
        """Replace the persisted representation of one memory."""
        ...

