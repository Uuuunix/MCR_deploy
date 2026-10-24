"""Source-message persistence errors."""


class SourceStoreError(RuntimeError):
    """Base source store failure."""


class ImmutableSourceConflict(SourceStoreError):
    """An immutable message identity was reused with different content."""


class SourceImageError(SourceStoreError):
    """An image could not be materialized into source bytes."""
