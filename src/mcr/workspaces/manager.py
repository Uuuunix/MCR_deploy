"""Lazy, concurrency-safe workspace service resolution."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from mcr.workspaces.catalog import WorkspaceCatalog
from mcr.workspaces.models import WorkspaceRecord

T = TypeVar("T")
WorkspaceFactory = Callable[[WorkspaceRecord], T | Awaitable[T]]


class WorkspaceManager(Generic[T]):
    def __init__(self, catalog: WorkspaceCatalog, factory: WorkspaceFactory[T]) -> None:
        self.catalog = catalog
        self._factory = factory
        self._handles: dict[str, T] = {}
        self._lock = asyncio.Lock()

    def create(self, workspace_id: str) -> WorkspaceRecord:
        return self.catalog.create(workspace_id)

    def list(self) -> list[WorkspaceRecord]:
        return self.catalog.list()

    def exists(self, workspace_id: str) -> bool:
        return self.catalog.exists(workspace_id)

    async def open(self, workspace_id: str = "default") -> T:
        record = self.catalog.get(workspace_id)
        cached = self._handles.get(record.id)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._handles.get(record.id)
            if cached is not None:
                return cached
            value = self._factory(record)
            handle = await value if inspect.isawaitable(value) else value
            self._handles[record.id] = handle
            return handle

    async def aclose(self) -> None:
        async with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for handle in handles:
            closer = getattr(handle, "aclose", None)
            if closer is not None:
                value = closer()
                if inspect.isawaitable(value):
                    await value

