"""ASGI API implementing AML's Health, Add and Search endpoints."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

from deployment.aml.journal import RequestConflict
from deployment.aml.schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from deployment.aml.service import AMLService
from deployment.aml.settings import Settings


def create_app(settings: Settings | None = None, *, service: AMLService | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    owned_service = service is None
    if service is None:
        service = AMLService(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owned_service:
            await service.aclose()

    app = FastAPI(title="MCR AML multimodal adapter", lifespan=lifespan)

    def authenticate(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        expected = settings.memory_system_key
        if not expected:
            raise HTTPException(status_code=503, detail="AML_MEMORY_SYSTEM_KEY is not configured")
        supplied = x_api_key
        if supplied is None and authorization:
            prefix, _, credential = authorization.partition(" ")
            if prefix.lower() in {"bearer", "token"}:
                supplied = credential
            else:
                supplied = authorization
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid memory system key")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post(f"{settings.api_prefix}/add", response_model=AddResponse, dependencies=[Depends(authenticate)])
    async def add(request: AddRequest) -> AddResponse:
        try:
            return await service.add(request)
        except RequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/search", response_model=SearchResponse, dependencies=[Depends(authenticate)])
    async def search(request: SearchRequest) -> SearchResponse:
        try:
            return await service.search(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
