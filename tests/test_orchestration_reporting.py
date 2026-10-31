import json

import pandas as pd
import pytest

from datasleuth.checks import MissingValueCheck
from datasleuth.engine import (
    AuditContext,
    AuditOrchestrator,
    CheckRegistry,
    CheckStatus,
    Finding,
    Severity,
)
from datasleuth.reporting import export_html, export_json


class FailingCheck:
    check_id = "TEST-FAIL-001"
    category = "Test"

    def run(self, context):
        raise RuntimeError("intentional failure")


class CriticalCheck:
    check_id = "TEST-CRITICAL-001"
    category = "Test"

    def run(self, context):
        return [
            Finding(
                check_id=self.check_id,
                category=self.category,
                severity=Severity.CRITICAL,
                title="Critical test finding",
                description="A test finding.",
                column=None,
                evidence={"ok": True},
                recommendation="Investigate.",
            )
        ]


def test_orchestrator_runs_selected_checks_and_reports_skips():
    data = pd.DataFrame({"value": [1, None]})
    registry = CheckRegistry([MissingValueCheck(), CriticalCheck()])

    result = AuditOrchestrator(registry).run(
        AuditContext(data=data),
        selected_check_ids={"QUALITY-MISSING-001"},
        metadata={"dataset": "example"},
    )

    statuses = {check.check_id: check.status for check in result.check_results}
    assert statuses == {
        "QUALITY-MISSING-001": CheckStatus.PASSED,
        "TEST-CRITICAL-001": CheckStatus.SKIPPED,
    }
    assert result.findings[0].check_id == "QUALITY-MISSING-001"
    assert result.metadata["dataset"] == "example"


def test_orchestrator_isolates_failures_and_keeps_successful_findings():
    registry = CheckRegistry([FailingCheck(), CriticalCheck()])

    result = AuditOrchestrator(registry).run(AuditContext(pd.DataFrame({"x": [1]})))

    statuses = {check.check_id: check.status for check in result.check_results}
    assert statuses["TEST-FAIL-001"] == CheckStatus.FAILED
    assert statuses["TEST-CRITICAL-001"] == CheckStatus.PASSED
    assert result.check_results[0].error == "RuntimeError: intentional failure"
    assert result.findings[0].check_id == "TEST-CRITICAL-001"


def test_findings_are_sorted_by_severity_and_report_exports_include_evidence():
    registry = CheckRegistry([CriticalCheck(), MissingValueCheck()])
    result = AuditOrchestrator(registry).run(
        AuditContext(pd.DataFrame({"value": [None, 1]}))
    )

    payload = json.loads(export_json(result))
    html = export_html(result)

    assert payload["summary"]["finding_count"] == 2
    assert payload["findings"][0]["severity"] == "CRITICAL"
    assert payload["findings"][1]["evidence"]["missing_count"] == 1
    assert "DataSleuth Audit Report" in html
    assert "Critical test finding" in html
    assert "Missing values detected" in html


def test_registry_and_selection_validate_duplicates_and_unknown_ids():
    with pytest.raises(ValueError, match="Duplicate check_id"):
        CheckRegistry([CriticalCheck(), CriticalCheck()])
    with pytest.raises(ValueError, match="Unknown check IDs"):
        AuditOrchestrator(CheckRegistry([CriticalCheck()])).run(
            AuditContext(pd.DataFrame({"x": [1]})),
            selected_check_ids={"UNKNOWN"},
        )
