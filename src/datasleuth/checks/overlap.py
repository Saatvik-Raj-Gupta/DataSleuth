"""Train/test row overlap check."""

from datasleuth.engine import AuditContext, Finding, Severity


class TrainTestOverlapCheck:
    """Detect exact row overlap between configured train and test data."""

    check_id = "LEAKAGE-SPLIT-001"
    category = "Leakage"

    def run(self, context: AuditContext) -> list[Finding]:
        if context.train_data is None or context.test_data is None:
            return []
        columns = list(
            context.train_data.columns.intersection(context.test_data.columns)
        )
        if not columns:
            raise ValueError("Train and test data have no common columns.")
        train_rows = context.train_data[columns].astype("string")
        test_rows = context.test_data[columns].astype("string")
        overlap = len(
            test_rows.merge(train_rows.drop_duplicates(), how="inner", on=columns)
        )
        if not overlap:
            return []
        return [
            Finding(
                check_id=self.check_id,
                category=self.category,
                severity=Severity.CRITICAL,
                title="Exact train/test row overlap detected",
                description=f"{overlap} test rows also appear in the training data.",
                column=None,
                evidence={
                    "overlap_rows": overlap,
                    "train_rows": len(context.train_data),
                    "test_rows": len(context.test_data),
                    "comparison_columns": columns,
                },
                recommendation=(
                    "Remove or investigate overlapping rows before evaluating "
                    "generalization."
                ),
            )
        ]
