"""Shared interface for independently executable checks."""

from typing import Protocol

from datasleuth.engine.contracts import AuditContext, Finding


class BaseCheck(Protocol):
    """Protocol implemented by every audit check."""

    check_id: str
    category: str

    def run(self, context: AuditContext) -> list[Finding]:
        """Run this check against an audit context."""
