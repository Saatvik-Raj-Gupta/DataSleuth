import pandas as pd
import pytest

from datasleuth.checks import (
    AllowedValueCheck,
    CategoryConsistencyCheck,
    ConstantColumnCheck,
    ExactDuplicateCheck,
    KeyDuplicateCheck,
    MissingValueCheck,
    OutlierCheck,
)
from datasleuth.engine import AuditContext, Severity


def context(data: pd.DataFrame) -> AuditContext:
    return AuditContext(data=data)


def test_clean_data_produces_no_quality_findings():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10.0, 11.0, 12.0],
            "category": ["A", "B", "A"],
        }
    )

    checks = [
        MissingValueCheck(),
        ExactDuplicateCheck(),
        KeyDuplicateCheck(["id"]),
        ConstantColumnCheck(),
        OutlierCheck(),
        CategoryConsistencyCheck(),
        AllowedValueCheck({"category": {"A", "B"}}),
    ]

    assert all(check.run(context(data)) == [] for check in checks)


def test_missing_and_duplicate_checks_report_evidence():
    data = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "value": [10.0, 10.0, None],
        }
    )
    audit_context = context(data)

    missing = MissingValueCheck().run(audit_context)
    exact = ExactDuplicateCheck().run(audit_context)
    keyed = KeyDuplicateCheck(["id"]).run(audit_context)

    assert missing[0].column == "value"
    assert missing[0].evidence["missing_count"] == 1
    assert exact[0].evidence["duplicate_row_count"] == 1
    assert keyed[0].evidence["duplicate_row_count"] == 1


def test_constant_near_constant_and_category_checks_report_findings():
    data = pd.DataFrame(
        {
            "constant": ["x"] * 100,
            "near": ["x"] * 99 + ["y"],
            "category": ["North", " north ", "NORTH"] + ["South"] * 97,
        }
    )

    constant_findings = ConstantColumnCheck().run(context(data))
    category_findings = CategoryConsistencyCheck().run(context(data))

    assert {finding.column for finding in constant_findings} == {"constant", "near"}
    assert constant_findings[0].severity == Severity.WARNING
    assert any(finding.column == "category" for finding in category_findings)


def test_outlier_check_labels_values_as_candidates_for_review():
    data = pd.DataFrame({"value": [10, 11, 12, 13, 1000]})

    findings = OutlierCheck(method="both").run(context(data))

    assert len(findings) == 1
    assert findings[0].column == "value"
    assert "candidates for review" in findings[0].description
    assert findings[0].evidence["outlier_counts"]["iqr"] == 1


def test_allowed_values_report_invalid_non_missing_values():
    data = pd.DataFrame({"status": ["new", "done", None, "unknown"]})

    findings = AllowedValueCheck({"status": ["new", "done"]}).run(context(data))

    assert findings[0].evidence["invalid_count"] == 1
    assert findings[0].evidence["invalid_values"] == ["unknown"]


def test_checks_handle_empty_and_all_null_data():
    data = pd.DataFrame({"empty": [None, None], "nulls": [None, 1]})
    empty_data = pd.DataFrame({"empty": pd.Series(dtype="float64")})

    assert {
        finding.column for finding in MissingValueCheck().run(context(data))
    } == {"empty", "nulls"}
    assert MissingValueCheck().run(context(empty_data)) == []
    assert ExactDuplicateCheck().run(context(data)) == []
    assert {
        finding.column for finding in ConstantColumnCheck().run(context(data))
    } == {"nulls"}
    assert OutlierCheck().run(context(data)) == []
    assert CategoryConsistencyCheck().run(context(data)) == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MissingValueCheck(threshold=1.1),
        lambda: KeyDuplicateCheck([]),
        lambda: ConstantColumnCheck(near_constant_threshold=0),
        lambda: OutlierCheck(method="invalid"),
    ],
)
def test_invalid_check_configuration_is_explicit(factory):
    with pytest.raises(ValueError):
        factory()


def test_checks_reject_missing_configured_columns():
    audit_context = context(pd.DataFrame({"id": [1]}))

    with pytest.raises(ValueError, match="Key columns"):
        KeyDuplicateCheck(["missing"]).run(audit_context)
    with pytest.raises(ValueError, match="Allowed-value columns"):
        AllowedValueCheck({"missing": ["x"]}).run(audit_context)
