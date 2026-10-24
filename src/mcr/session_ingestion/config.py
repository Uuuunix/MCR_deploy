"""Configuration for automatic session ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class SessionIngestionConfig:
    recent_window_size: int = 8
    lease_seconds: float = 300.0
    heartbeat_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.recent_window_size < 0:
            raise ValueError("recent_window_size must be non-negative")
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if not 0 < self.heartbeat_seconds < self.lease_seconds:
            raise ValueError("heartbeat_seconds must be positive and shorter than the lease")

    @classmethod
    def from_environment(cls) -> "SessionIngestionConfig":
        return cls(
            recent_window_size=int(os.getenv("MCR_RECENT_WINDOW_SIZE", "8")),
            lease_seconds=float(os.getenv("MCR_SUBMISSION_LEASE_SECONDS", "300")),
            heartbeat_seconds=float(
                os.getenv("MCR_SUBMISSION_HEARTBEAT_SECONDS", "30")
            ),
        )
