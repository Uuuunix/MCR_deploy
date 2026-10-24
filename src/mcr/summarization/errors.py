"""Summary feature errors."""


class SummaryError(RuntimeError):
    """Base error for session summary generation and coordination."""


class SummaryResponseError(SummaryError):
    """The provider returned an unusable summary."""
