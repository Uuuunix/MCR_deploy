"""Normalize summary model responses."""

from __future__ import annotations

import re

from mcr.summarization.errors import SummaryResponseError


_FENCE = re.compile(r"^```(?:text|markdown|md)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def parse_summary_response(raw: str) -> str:
    value = raw.strip()
    match = _FENCE.match(value)
    if match:
        value = match.group(1).strip()
    if not value:
        raise SummaryResponseError("summary model returned an empty response")
    return value
