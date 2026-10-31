"""Missing-value quality checks."""

from datasleuth.engine import AuditContext, Finding, Severity


class MissingValueCheck:
    """Report columns whose missingness meets a configurable threshold."""

    check_id = "QUALITY-MISSING-001"
    category = "Data Quality"

    def __init__(self, threshold: float = 0.0) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        self.threshold = threshold

    def run(self, context: AuditContext) -> list[Finding]:
        findings = []
        for column in context.data.columns:
            series = context.data[column]
            missing_count = int(series.isna().sum())
            percentage = missing_count / len(series) if len(series) else 0.0
            if missing_count and percentage >= self.threshold:
                findings.append(
                    Finding(
                        check_id=self.check_id,
                        category=self.category,
                        severity=Severity.WARNING,
                        title="Missing values detected",
                        description=(
                            f"The column contains {percentage:.2%} missing values."
                        ),
                        column=str(column),
                        evidence={
                            "missing_count": missing_count,
                            "total_rows": len(series),
                            "missing_percentage": round(percentage * 100, 4),
                        },
                        recommendation=(
                            "Investigate the source of missingness before choosing "
                            "an imputation strategy."
                        ),
                    )
                )
        return findings

