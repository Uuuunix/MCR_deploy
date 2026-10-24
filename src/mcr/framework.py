"""Small public facade over workspace-bound Agent lifecycles."""

from __future__ import annotations

import asyncio

from mcr.agent_lifecycle import (
    AfterGenerationRequest,
    AfterGenerationResult,
    BeforeGenerationRequest,
    BeforeGenerationResult,
    ExchangeIngestionResult,
)
from mcr.workspaces import WorkspaceManager


class MCR:
    def __init__(self, workspaces: WorkspaceManager) -> None:
        self.workspaces = workspaces

    async def before_generation(
        self,
        request: BeforeGenerationRequest,
        *,
        workspace_id: str = "default",
    ) -> BeforeGenerationResult:
        workspace = await self.workspaces.open(workspace_id)
        return await workspace.before_generation(request)

    async def after_generation(
        self,
        request: AfterGenerationRequest,
        *,
        workspace_id: str = "default",
    ) -> AfterGenerationResult:
        workspace = await self.workspaces.open(workspace_id)
        return await workspace.after_generation(request)

    async def ingest_exchange(
        self, *, workspace_id: str = "default", **kwargs
    ) -> ExchangeIngestionResult:
        workspace = await self.workspaces.open(workspace_id)
        return await workspace.ingest_exchange(**kwargs)

    def ingest_exchange_sync(
        self, *, workspace_id: str = "default", **kwargs
    ) -> ExchangeIngestionResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.ingest_exchange(workspace_id=workspace_id, **kwargs)
            )
        raise RuntimeError(
            "ingest_exchange_sync cannot run inside an active event loop; "
            "await ingest_exchange instead"
        )

