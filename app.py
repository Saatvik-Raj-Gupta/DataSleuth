"""Streamlit entry point for the DataSleuth audit application."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

from datasleuth.checks import (
    AllowedValueCheck,
    CategoricalAssociationCheck,
    CategoryConsistencyCheck,
    ConstantColumnCheck,
    DistributionShiftCheck,
    ExactDuplicateCheck,
    ExactTargetCopyCheck,
    IdLikeCheck,
    ImbalanceCheck,
    KeyDuplicateCheck,
    MissingValueCheck,
    MutualInformationCheck,
    NumericalAssociationCheck,
    OutlierCheck,
    TemporalContextCheck,
    TrainTestOverlapCheck,
)
from datasleuth.engine import (
    AuditContext,
    AuditOrchestrator,
    AuditResult,
    CheckRegistry,
    CheckStatus,
    Severity,
)
from datasleuth.ingestion import CsvLoadConfig, load_csv
from datasleuth.ingestion.errors import IngestionError
from datasleuth.profiling import DatasetProfile, profile_dataframe
from datasleuth.reporting import (
    categorical_distribution_figure,
    export_html,
    export_json,
    numerical_distribution_figure,
)

QUALITY_IDS = {
    "QUALITY-MISSING-001",
    "QUALITY-DUPLICATES-001",
    "QUALITY-DUPLICATES-002",
    "QUALITY-CONSTANT-001",
    "QUALITY-OUTLIERS-001",
    "QUALITY-CATEGORIES-001",
    "QUALITY-VALUES-001",
}
TARGET_IDS = {
    "LEAKAGE-TARGET-001",
    "LEAKAGE-NUMERIC-001",
    "LEAKAGE-MUTUAL-INFO-001",
    "LEAKAGE-CATEGORICAL-001",
    "DISTRIBUTION-IMBALANCE-001",
}
SPLIT_IDS = {"LEAKAGE-SPLIT-001", "DISTRIBUTION-SHIFT-001"}


def build_checks(
    *,
    missing_threshold: float,
    near_constant_threshold: float,
    outlier_method: str,
    numerical_threshold: float,
    categorical_threshold: float,
    mutual_information_threshold: float,
    id_uniqueness_threshold: float,
    psi_threshold: float,
    key_columns: list[str],
    allowed_values: dict[str, set[str]],
    event_time_column: str | None,
) -> list[Any]:
    """Build configured checks without implementing statistical logic."""
    checks: list[Any] = [
        MissingValueCheck(threshold=missing_threshold),
        ExactDuplicateCheck(),
        ConstantColumnCheck(near_constant_threshold=near_constant_threshold),
        OutlierCheck(method=outlier_method),
        CategoryConsistencyCheck(),
        AllowedValueCheck(allowed_values),
        ExactTargetCopyCheck(),
        NumericalAssociationCheck(threshold=numerical_threshold),
        MutualInformationCheck(threshold=mutual_information_threshold),
        CategoricalAssociationCheck(threshold=categorical_threshold),
        IdLikeCheck(uniqueness_threshold=id_uniqueness_threshold),
        ImbalanceCheck(),
        DistributionShiftCheck(psi_threshold=psi_threshold),
        TrainTestOverlapCheck(),
    ]
    if key_columns:
        checks.append(KeyDuplicateCheck(key_columns))
    if event_time_column:
        checks.append(TemporalContextCheck(event_time_column))
    return checks


def selected_check_ids(
    checks: list[Any],
    *,
    has_target: bool,
    has_split: bool,
) -> set[str]:
    """Select checks whose required UI configuration is available."""
    unavailable = set()
    if not has_target:
        unavailable |= TARGET_IDS
    if not has_split:
        unavailable |= SPLIT_IDS
    return {check.check_id for check in checks if check.check_id not in unavailable}


def _read_upload(upload: Any, *, max_bytes: int = 100 * 1024 * 1024) -> pd.DataFrame:
    return load_csv(
        BytesIO(upload.getvalue()),
        config=CsvLoadConfig(max_bytes=max_bytes),
    )


def _render_profile(profile: DatasetProfile) -> None:
    left, middle, right = st.columns(3)
    left.metric("Rows", profile.row_count)
    middle.metric("Columns", profile.column_count)
    right.metric("Memory", f"{profile.memory_bytes:,} bytes")
    st.dataframe(
        pd.DataFrame([column.to_dict() for column in profile.column_profiles]),
        use_container_width=True,
    )


def _render_statuses(result: AuditResult) -> None:
    st.subheader("Check statuses")
    status_columns = st.columns(3)
    counts = {
        status: sum(check.status == status for check in result.check_results)
        for status in CheckStatus
    }
    for column, status in zip(status_columns, CheckStatus, strict=True):
        column.metric(status.value.title(), counts[status])
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "check_id": check.check_id,
                    "category": check.category,
                    "status": check.status.value,
                    "error": check.error or "",
                }
                for check in result.check_results
            ]
        ),
        use_container_width=True,
    )


def _render_findings(result: AuditResult) -> None:
    st.subheader("Findings")
    categories = sorted({finding.category for finding in result.findings})
    severities = [severity.value for severity in Severity]
    chosen_categories = st.multiselect("Category", categories, default=categories)
    chosen_severities = st.multiselect(
        "Severity", severities, default=severities
    )
    findings = [
        finding
        for finding in result.findings
        if finding.category in chosen_categories
        and finding.severity.value in chosen_severities
    ]
    for finding in findings:
        with st.expander(
            f"{finding.severity.value}: {finding.title} "
            f"({finding.column or 'dataset'})"
        ):
            st.write(finding.description)
            st.json(
                {
                    "check_id": finding.check_id,
                    "category": finding.category,
                    "column": finding.column,
                    "evidence": finding.evidence,
                    "recommendation": finding.recommendation,
                }
            )
    if not findings:
        st.info("No findings match the selected filters.")


def _render_distribution_charts(
    train_data: pd.DataFrame | None,
    test_data: pd.DataFrame | None,
) -> None:
    if train_data is None or test_data is None:
        st.info("Upload both train and test CSV files to view distributions.")
        return
    numerical = [
        column
        for column in train_data.columns.intersection(test_data.columns)
        if pd.api.types.is_numeric_dtype(train_data[column])
        and pd.api.types.is_numeric_dtype(test_data[column])
    ]
    categorical = [
        column
        for column in train_data.columns.intersection(test_data.columns)
        if (
            pd.api.types.is_object_dtype(train_data[column])
            or pd.api.types.is_string_dtype(train_data[column])
        )
        and (
            pd.api.types.is_object_dtype(test_data[column])
            or pd.api.types.is_string_dtype(test_data[column])
        )
    ]
    if numerical:
        selected = st.selectbox("Numerical column", numerical)
        st.plotly_chart(
            numerical_distribution_figure(
                train_data[selected].dropna(),
                test_data[selected].dropna(),
                column=selected,
            ),
            use_container_width=True,
        )
    if categorical:
        selected = st.selectbox("Categorical column", categorical)
        st.plotly_chart(
            categorical_distribution_figure(
                train_data[selected].dropna(),
                test_data[selected].dropna(),
                column=selected,
            ),
            use_container_width=True,
        )
    if not numerical and not categorical:
        st.info("No compatible common columns are available for visualization.")


def main() -> None:
    """Render the DataSleuth Streamlit application."""
    st.set_page_config(page_title="DataSleuth", page_icon="🔎", layout="wide")
    st.title("DataSleuth")
    st.caption("Evidence-backed dataset quality and leakage auditing")

    with st.sidebar:
        st.header("Upload and configuration")
        data_upload = st.file_uploader("Primary CSV", type=["csv"])
        train_upload = st.file_uploader("Reference/train CSV", type=["csv"])
        test_upload = st.file_uploader("Comparison/test CSV", type=["csv"])
        missing_threshold = st.slider("Missing-value threshold", 0.0, 1.0, 0.0)
        near_constant_threshold = st.slider(
            "Near-constant threshold", 0.5, 1.0, 0.99
        )
        outlier_method = st.selectbox("Outlier method", ["iqr", "zscore", "both"])
        psi_threshold = st.number_input("PSI threshold", min_value=0.01, value=0.2)
        key_text = st.text_input("Duplicate key columns (comma-separated)")
        event_time_column = st.text_input("Event-time column (optional)")
        run_audit = st.button("Run audit", type="primary", use_container_width=True)

    if not data_upload:
        st.info("Upload a primary CSV file to begin.")
        return

    try:
        data = _read_upload(data_upload)
        train_data = _read_upload(train_upload) if train_upload else None
        test_data = _read_upload(test_upload) if test_upload else None
    except IngestionError as exc:
        st.error(str(exc))
        return

    st.header("Dataset overview")
    _render_profile(profile_dataframe(data))
    target_options = ["None", *[str(column) for column in data.columns]]
    target = st.selectbox("Target column", target_options)
    task_type = st.selectbox("Task type", ["classification", "regression"])

    with st.expander("Distribution visualizations"):
        _render_distribution_charts(train_data, test_data)

    if not run_audit:
        st.info("Configure the audit and select Run audit.")
        return

    key_columns = [column.strip() for column in key_text.split(",") if column.strip()]
    checks = build_checks(
        missing_threshold=missing_threshold,
        near_constant_threshold=near_constant_threshold,
        outlier_method=outlier_method,
        numerical_threshold=0.95,
        categorical_threshold=0.9,
        mutual_information_threshold=0.5,
        id_uniqueness_threshold=0.98,
        psi_threshold=psi_threshold,
        key_columns=key_columns,
        allowed_values={},
        event_time_column=event_time_column or None,
    )
    has_target = target != "None"
    has_split = train_data is not None and test_data is not None
    context = AuditContext(
        data=data,
        target=target if has_target else None,
        task_type=task_type if has_target else None,
        train_data=train_data,
        test_data=test_data,
        metadata={"source": data_upload.name},
    )
    result = AuditOrchestrator(CheckRegistry(checks)).run(
        context,
        selected_check_ids=selected_check_ids(
            checks, has_target=has_target, has_split=has_split
        ),
        metadata={
            "source": data_upload.name,
            "rows": len(data),
            "columns": len(data.columns),
            "target": context.target,
            "task_type": context.task_type,
        },
    )
    st.session_state["audit_result"] = result

    st.header("Audit results")
    _render_statuses(result)
    _render_findings(result)
    st.download_button(
        "Download JSON report",
        export_json(result),
        file_name="datasleuth-report.json",
        mime="application/json",
    )
    st.download_button(
        "Download HTML report",
        export_html(result),
        file_name="datasleuth-report.html",
        mime="text/html",
    )


if __name__ == "__main__":
    main()
