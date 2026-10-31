import json

import numpy as np
import pandas as pd

from datasleuth.engine import AuditContext, Finding, Severity


def test_finding_serializes_numpy_and_timestamp_evidence():
    finding = Finding(
        check_id="QUALITY-MISSING-001",
        category="Data Quality",
        severity=Severity.WARNING,
        title="High missing-value percentage",
        description="A column contains missing values.",
        column="annual_income",
        evidence={
            "missing_count": np.int64(380),
            "observed_at": pd.Timestamp("2026-01-01"),
        },
        recommendation="Investigate the source of missingness.",
    )

    payload = json.loads(finding.to_json())

    assert payload["severity"] == "WARNING"
    assert payload["evidence"]["missing_count"] == 380
    assert payload["evidence"]["observed_at"] == "2026-01-01T00:00:00"
    assert finding.to_dict() == payload


def test_audit_context_keeps_shared_optional_inputs():
    data = pd.DataFrame({"target": [0, 1]})
    context = AuditContext(
        data=data,
        target="target",
        task_type="classification",
        config={"threshold": 0.5},
    )

    assert context.data is data
    assert context.target == "target"
    assert context.task_type == "classification"
    assert context.config == {"threshold": 0.5}
