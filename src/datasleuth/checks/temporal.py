"""Optional temporal context warnings."""

from datasleuth.engine import AuditContext, Finding, Severity


class TemporalContextCheck:
    """Warn when a configured event-time column is absent from context metadata."""

    check_id = "LEAKAGE-TEMPORAL-001"
    category = "Leakage"

    def __init__(self, event_time_column: str) -> None:
        if not event_time_column:
            raise ValueError("event_time_column is required.")
        self.event_time_column = event_time_column

    def run(self, context: AuditContext) -> list[Finding]:
        if self.event_time_column not in context.data:
            raise ValueError(
                f"Event-time column is not present: {self.event_time_column}."
            )
        metadata = context.metadata or {}
        if metadata.get("prediction_time") is not None:
            return []
        return [
            Finding(
                check_id=self.check_id,
                category=self.category,
                severity=Severity.INFO,
                title="Prediction-time context is not configured",
                description=(
                    "An event-time column exists, but prediction-time context was "
                    "not supplied to assess temporal leakage."
                ),
                column=self.event_time_column,
                evidence={"event_time_column": self.event_time_column},
                recommendation=(
                    "Supply prediction_time metadata and verify each feature was "
                    "available before the prediction timestamp."
                ),
            )
        ]

