"""Constant and near-constant column checks."""

from datasleuth.engine import AuditContext, Finding, Severity


class ConstantColumnCheck:
    """Report constant and configurable near-constant columns."""

    check_id = "QUALITY-CONSTANT-001"
    category = "Data Quality"

    def __init__(self, near_constant_threshold: float = 0.99) -> None:
        if not 0 < near_constant_threshold <= 1:
            raise ValueError(
                "near_constant_threshold must be greater than 0 and at most 1."
            )
        self.near_constant_threshold = near_constant_threshold

    def run(self, context: AuditContext) -> list[Finding]:
        findings = []
        for column in context.data.columns:
            series = context.data[column].dropna()
            if series.empty:
                continue
            proportions = series.value_counts(normalize=True)
            dominant_fraction = float(proportions.iloc[0])
            if dominant_fraction == 1:
                severity = Severity.WARNING
                title = "Constant column detected"
                description = "All non-missing values in the column are identical."
            elif dominant_fraction >= self.near_constant_threshold:
                severity = Severity.INFO
                title = "Near-constant column detected"
                description = (
                    f"The most common value represents {dominant_fraction:.2%} "
                    "of non-missing values."
                )
            else:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=severity,
                    title=title,
                    description=description,
                    column=str(column),
                    evidence={
                        "non_missing_count": len(series),
                        "dominant_fraction": round(dominant_fraction, 6),
                        "distinct_non_missing_values": int(series.nunique()),
                    },
                    recommendation=(
                        "Confirm whether this feature provides useful variation "
                        "for the intended analysis."
                    ),
                )
            )
        return findings
