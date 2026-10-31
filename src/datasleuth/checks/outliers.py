"""Robust outlier candidate screening."""

import pandas as pd

from datasleuth.engine import AuditContext, Finding, Severity


class OutlierCheck:
    """Report numeric values outside IQR and/or Z-score bounds."""

    check_id = "QUALITY-OUTLIERS-001"
    category = "Data Quality"

    def __init__(
        self,
        *,
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
    ) -> None:
        if method not in {"iqr", "zscore", "both"}:
            raise ValueError("method must be 'iqr', 'zscore', or 'both'.")
        if iqr_multiplier <= 0 or z_threshold <= 0:
            raise ValueError("outlier thresholds must be greater than zero.")
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.z_threshold = z_threshold

    def run(self, context: AuditContext) -> list[Finding]:
        findings = []
        for column in context.data.select_dtypes(include="number").columns:
            series = context.data[column].dropna()
            if len(series) < 3 or series.nunique() < 2:
                continue
            masks = {}
            if self.method in {"iqr", "both"}:
                q1, q3 = series.quantile([0.25, 0.75])
                spread = q3 - q1
                masks["iqr"] = (series < q1 - self.iqr_multiplier * spread) | (
                    series > q3 + self.iqr_multiplier * spread
                )
            if self.method in {"zscore", "both"}:
                standard_deviation = series.std()
                if standard_deviation:
                    masks["zscore"] = (
                        (series - series.mean()).abs() / standard_deviation
                    ) > self.z_threshold
                else:
                    masks["zscore"] = pd.Series(False, index=series.index)
            outlier_counts = {name: int(mask.sum()) for name, mask in masks.items()}
            total_outliers = int((pd.concat(masks, axis=1).any(axis=1)).sum())
            if not total_outliers:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=Severity.WARNING,
                    title="Potential outliers detected",
                    description=(
                        f"{total_outliers} values are candidates for review; "
                        "they are not necessarily errors."
                    ),
                    column=str(column),
                    evidence={
                        "method": self.method,
                        "outlier_counts": outlier_counts,
                        "non_missing_count": len(series),
                    },
                    recommendation=(
                        "Review flagged values against domain rules and source "
                        "records before changing or removing them."
                    ),
                )
            )
        return findings

