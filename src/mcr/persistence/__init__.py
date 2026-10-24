"""Persistence contracts independent from any database."""

from mcr.persistence.errors import (
    PersistenceConfigurationError,
    PersistenceError,
    PersistenceWriteError,
)
from mcr.persistence.repository import MemoryRepository

__all__ = [
    "MemoryRepository",
    "PersistenceConfigurationError",
    "PersistenceError",
    "PersistenceWriteError",
]

