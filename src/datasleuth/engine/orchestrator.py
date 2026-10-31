"""Check registration and fault-isolated audit orchestration."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from datasleuth.checks.base import BaseCheck
from datasleuth.engine.contracts import AuditContext, Finding, Severity

logger = logging.getLogger(__name__)


class CheckStatus(StrEnum):
    """Outcome of one registered check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CheckResult:
    """Outcome and findings for one check."""

    check_id: str
    category: str
    status: CheckStatus
    findings: tuple[Finding, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible result data."""
        return {
            "check_id": self.check_id,
            "category": self.category,
            "status": self.status.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "error": self.error,
        }


@dataclass(frozen=True)
class AuditResult:
    """Complete result from one orchestrator run."""

    findings: tuple[Finding, ...]
    check_results: tuple[CheckResult, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical report dictionary."""
        status_counts = {
            status.value: sum(
                result.status == status for result in self.check_results
            )
            for status in CheckStatus
        }
        severity_counts = {
            severity.value: sum(
                finding.severity == severity for finding in self.findings
            )
            for severity in Severity
        }
        return {
            "version": "0.1.0",
            "metadata": dict(self.metadata),
            "summary": {
                "finding_count": len(self.findings),
                "check_counts": status_counts,
                "severity_counts": severity_counts,
            },
            "checks": [result.to_dict() for result in self.check_results],
            "findings": [finding.to_dict() for finding in self.findings],
        }


class CheckRegistry:
    """Validated collection of checks selected for an audit."""

    def __init__(self, checks: Iterable[BaseCheck] = ()) -> None:
        self._checks: dict[str, BaseCheck] = {}
        for check in checks:
            self.register(check)

    def register(self, check: BaseCheck) -> None:
        """Register one check, rejecting duplicate or malformed identifiers."""
        check_id = getattr(check, "check_id", None)
        category = getattr(check, "category", None)
        run = getattr(check, "run", None)
        if not isinstance(check_id, str) or not check_id:
            raise ValueError("Every check must define a non-empty check_id.")
        if not isinstance(category, str) or not category:
            raise ValueError(f"Check {check_id!r} must define a category.")
        if not callable(run):
            raise ValueError(f"Check {check_id!r} must define a callable run method.")
        if check_id in self._checks:
            raise ValueError(f"Duplicate check_id registered: {check_id}.")
        self._checks[check_id] = check

    def checks(self) -> tuple[BaseCheck, ...]:
        """Return checks in registration order."""
        return tuple(self._checks.values())


class AuditOrchestrator:
    """Run selected checks while isolating individual failures."""

    def __init__(self, registry: CheckRegistry) -> None:
        self.registry = registry

    def run(
        self,
        context: AuditContext,
        *,
        selected_check_ids: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditResult:
        """Run registered checks and preserve failures as explicit statuses."""
        selected = (
            set(selected_check_ids) if selected_check_ids is not None else None
        )
        registered_ids = {check.check_id for check in self.registry.checks()}
        unknown = selected - registered_ids if selected is not None else set()
        if unknown:
            raise ValueError(f"Unknown check IDs selected: {sorted(unknown)}.")

        results = []
        all_findings: list[Finding] = []
        for check in self.registry.checks():
            if selected is not None and check.check_id not in selected:
                results.append(
                    CheckResult(
                        check_id=check.check_id,
                        category=check.category,
                        status=CheckStatus.SKIPPED,
                    )
                )
                continue
            try:
                findings = tuple(check.run(context))
                _validate_findings(check, findings)
            except Exception as exc:
                logger.exception("Check %s failed", check.check_id)
                results.append(
                    CheckResult(
                        check_id=check.check_id,
                        category=check.category,
                        status=CheckStatus.FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            normalized = tuple(sorted(findings, key=_finding_sort_key))
            all_findings.extend(normalized)
            results.append(
                CheckResult(
                    check_id=check.check_id,
                    category=check.category,
                    status=CheckStatus.PASSED,
                    findings=normalized,
                )
            )

        sorted_findings = tuple(sorted(all_findings, key=_finding_sort_key))
        return AuditResult(
            findings=sorted_findings,
            check_results=tuple(results),
            metadata=dict(metadata or {}),
        )


def _validate_findings(check: BaseCheck, findings: tuple[Finding, ...]) -> None:
    if not isinstance(findings, tuple):
        raise TypeError(
            f"Check {check.check_id} must return a list of Finding objects."
        )
    if not all(isinstance(finding, Finding) for finding in findings):
        raise TypeError(f"Check {check.check_id} returned an invalid finding.")


def _finding_sort_key(finding: Finding) -> tuple[int, str, str, str]:
    severity_rank = {
        Severity.CRITICAL: 0,
        Severity.WARNING: 1,
        Severity.INFO: 2,
    }
    return (
        severity_rank[finding.severity],
        finding.category,
        finding.check_id,
        finding.column or "",
    )
