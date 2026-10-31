"""Categorical whitespace and casing consistency checks."""

from collections import defaultdict

from datasleuth.engine import AuditContext, Finding, Severity


class CategoryConsistencyCheck:
    """Report categories that differ only by whitespace or casing."""

    check_id = "QUALITY-CATEGORIES-001"
    category = "Data Quality"

    def run(self, context: AuditContext) -> list[Finding]:
        findings = []
        for column in context.data.select_dtypes(
            include=["object", "string", "category"]
        ).columns:
            series = context.data[column].dropna()
            variants: dict[str, set[str]] = defaultdict(set)
            for value in series:
                original = str(value)
                variants[original.strip().casefold()].add(original)
            inconsistent = {
                normalized: sorted(values)
                for normalized, values in variants.items()
                if len(values) > 1
            }
            if not inconsistent:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=Severity.WARNING,
                    title="Inconsistent categorical values detected",
                    description=(
                        "Some categories differ only by whitespace or letter case."
                    ),
                    column=str(column),
                    evidence={"normalized_variants": inconsistent},
                    recommendation=(
                        "Confirm the intended category vocabulary before applying "
                        "an explicit normalization rule."
                    ),
                )
            )
        return findings
