"""Stable errors for memory extraction."""

from __future__ import annotations

from typing import Any


class ExtractionError(RuntimeError):
    """Base error for extraction failures."""


class ExtractionResponseError(ExtractionError):
    """The LLM returned invalid or unsafe extraction output."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        issues: list[dict[str, Any]] | None = None,
        response_diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        self.message = message
        self.raw_response = raw_response
        self.issues = issues or []
        self.response_diagnostics = response_diagnostics or []
        issue_text = "; ".join(
            f"{'.'.join(str(part) for part in issue.get('loc', ())) or '<root>'}: "
            f"{issue.get('msg', 'invalid value')}"
            for issue in self.issues[:8]
        )
        super().__init__(f"{message}: {issue_text}" if issue_text else message)

    def debug_detail(self) -> dict[str, Any]:
        return {
            "error": self.message,
            "issues": self.issues,
            "raw_response": self.raw_response,
            "response_diagnostics": self.response_diagnostics,
        }

