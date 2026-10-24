"""Public AML Add/Search wire contract, independent of MCR's debug API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class AMLTextPart(BaseModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class AMLImageURL(BaseModel):
    url: str = Field(min_length=1)


class AMLImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: AMLImageURL


AMLPart = Annotated[AMLTextPart | AMLImagePart, Field(discriminator="type")]


class AMLMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[AMLPart]
    timestamp: int | None = None

    @model_validator(mode="after")
    def nonempty(self) -> "AMLMessage":
        if isinstance(self.content, str) and not self.content.strip():
            raise ValueError("message content cannot be empty")
        if isinstance(self.content, list) and not self.content:
            raise ValueError("message content array cannot be empty")
        return self


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: list[AMLMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str | list[AMLPart]
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1)

    @model_validator(mode="after")
    def nonempty(self) -> "SearchRequest":
        if isinstance(self.query, str) and not self.query.strip():
            raise ValueError("query cannot be empty")
        if isinstance(self.query, list) and not self.query:
            raise ValueError("query content array cannot be empty")
        return self


class SearchMemory(BaseModel):
    id: str
    content: str | list[AMLTextPart | AMLImagePart]
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    data: list[SearchMemory]
