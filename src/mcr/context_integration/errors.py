"""Memory context integration errors."""


class MemoryContextIntegrationError(ValueError):
    """Raised when an integrator returns an invalid result."""


class CriticalIntegrationError(MemoryContextIntegrationError):
    """Evidence cannot be safely handed to the answer model."""


class SourceImageUnavailable(CriticalIntegrationError):
    """A linked original image is missing or unreadable."""

