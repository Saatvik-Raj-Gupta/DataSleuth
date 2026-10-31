"""ID-like feature heuristic."""

from datasleuth.engine import AuditContext, Finding, Severity


class IdLikeCheck:
    """Report columns whose non-missing values are almost all unique."""

    check_id = "LEAKAGE-IDLIKE-001"
    category = "Leakage"

    def __init__(self, uniqueness_threshold: float = 0.98) -> None:
        if not 0 < uniqueness_threshold <= 1:
            raise ValueError(
                "uniqueness_threshold must be greater than 0 and at most 1."
            )
        self.uniqueness_threshold = uniqueness_threshold

    def run(self, context: AuditContext) -> list[Finding]:
        findings = []
        for column in context.data.columns:
            if column == context.target:
                continue
            values = context.data[column].dropna()
            if len(values) < 3:
                continue
            uniqueness = values.nunique() / len(values)
            if uniqueness < self.uniqueness_threshold:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=Severity.INFO,
                    title="ID-like feature detected",
                    description=(
                        "The feature is nearly unique per row and may identify "
                        "records rather than represent a generalizable signal."
                    ),
                    column=str(column),
                    evidence={
                        "unique_non_missing": int(values.nunique()),
                        "non_missing_count": len(values),
                        "uniqueness_ratio": uniqueness,
                        "threshold": self.uniqueness_threshold,
                    },
                    recommendation=(
                        "Confirm the feature's intended role and availability; "
                        "ID-like appearance alone is not leakage."
                    ),
                )
            )
        return findings
