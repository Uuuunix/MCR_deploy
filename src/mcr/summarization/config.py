"""Configuration for incremental session summaries."""

from pydantic import Field

from mcr.memory.common import DomainModel


class SummaryConfig(DomainModel):
    batch_size: int = Field(default=5, gt=0)
    poll_seconds: float = Field(default=1.0, gt=0)
    max_attempts: int = Field(default=3, gt=0)
    max_tokens: int = Field(default=1024, gt=0)
