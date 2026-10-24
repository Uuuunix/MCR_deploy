"""Stable retrieval exceptions."""


class RetrievalError(RuntimeError):
    """Base class for retrieval failures."""


class RetrievalConfigurationError(RetrievalError):
    """The vector space or retrieval configuration is incompatible."""


class RetrievalDataError(RetrievalError):
    """Persisted retrieval data is malformed."""


class RetrievalRequestError(RetrievalError):
    """The vector backend request failed."""
