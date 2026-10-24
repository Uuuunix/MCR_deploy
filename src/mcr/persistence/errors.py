"""Stable persistence-layer exceptions."""


class PersistenceError(RuntimeError):
    """Base class for failures at the persistence boundary."""


class PersistenceConfigurationError(PersistenceError):
    """The configured backend is incompatible with MCR's requirements."""


class PersistenceWriteError(PersistenceError):
    """A backend failed while saving a valid memory."""

