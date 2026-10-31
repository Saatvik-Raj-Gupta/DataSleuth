"""Exact and selected-key duplicate checks."""

from collections.abc import Sequence

from datasleuth.engine import AuditContext, Finding, Severity


class ExactDuplicateCheck:
    """Report duplicate rows across all columns."""

    check_id = "QUALITY-DUPLICATES-001"
    category = "Data Quality"

    def run(self, context: AuditContext) -> list[Finding]:
        duplicate_count = int(context.data.duplicated(keep="first").sum())
        if not duplicate_count:
            return []
        return [
            Finding(
                check_id=self.check_id,
                category=self.category,
                severity=Severity.WARNING,
                title="Exact duplicate rows detected",
                description=f"{duplicate_count} duplicate rows were found.",
                column=None,
                evidence={
                    "duplicate_row_count": duplicate_count,
                    "total_rows": len(context.data),
                },
                recommendation=(
                    "Confirm whether duplicate records are expected before "
                    "deduplicating."
                ),
            )
        ]


class KeyDuplicateCheck:
    """Report duplicate records for user-selected key columns."""

    check_id = "QUALITY-DUPLICATES-002"
    category = "Data Quality"

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        if not self.columns:
            raise ValueError("At least one key column is required.")

    def run(self, context: AuditContext) -> list[Finding]:
        missing = [column for column in self.columns if column not in context.data]
        if missing:
            raise ValueError(f"Key columns are not present: {missing}.")
        duplicate_count = int(
            context.data.duplicated(subset=list(self.columns), keep="first").sum()
        )
        if not duplicate_count:
            return []
        return [
            Finding(
                check_id=self.check_id,
                category=self.category,
                severity=Severity.WARNING,
                title="Duplicate selected keys detected",
                description=(
                    f"{duplicate_count} rows duplicate the selected keys "
                    f"{', '.join(self.columns)}."
                ),
                column=", ".join(self.columns),
                evidence={
                    "duplicate_row_count": duplicate_count,
                    "key_columns": list(self.columns),
                    "total_rows": len(context.data),
                },
                recommendation=(
                    "Verify key uniqueness and determine whether repeated keys "
                    "represent valid observations."
                ),
            )
        ]

