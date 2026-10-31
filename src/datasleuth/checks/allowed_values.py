"""Optional explicit categorical allowed-value checks."""

from collections.abc import Iterable, Mapping

from datasleuth.engine import AuditContext, Finding, Severity


class AllowedValueCheck:
    """Report non-missing values outside configured per-column allow-lists."""

    check_id = "QUALITY-VALUES-001"
    category = "Data Quality"

    def __init__(self, allowed_values: Mapping[str, Iterable[object]]) -> None:
        self.allowed_values = {
            column: frozenset(values) for column, values in allowed_values.items()
        }

    def run(self, context: AuditContext) -> list[Finding]:
        findings = []
        missing_columns = [
            column for column in self.allowed_values if column not in context.data
        ]
        if missing_columns:
            raise ValueError(
                f"Allowed-value columns are not present: {missing_columns}."
            )
        for column, allowed in self.allowed_values.items():
            values = context.data[column].dropna()
            invalid = values[~values.isin(allowed)]
            if invalid.empty:
                continue
            values = sorted({str(value) for value in invalid.tolist()})
            findings.append(
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=Severity.WARNING,
                    title="Values outside allowed set detected",
                    description=(
                        f"{len(invalid)} values are not in the configured allow-list."
                    ),
                    column=str(column),
                    evidence={
                        "invalid_count": len(invalid),
                        "invalid_values": values,
                        "allowed_values": sorted(str(value) for value in allowed),
                    },
                    recommendation=(
                        "Validate the source vocabulary and update the explicit "
                        "allow-list only when the new value is valid."
                    ),
                )
            )
        return findings
