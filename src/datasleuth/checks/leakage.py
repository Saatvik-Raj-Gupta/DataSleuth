"""Target leakage screening checks."""

from collections.abc import Iterable
from math import sqrt

import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from datasleuth.engine import AuditContext, Finding, Severity


def _target_series(context: AuditContext) -> pd.Series | None:
    if context.target is None:
        return None
    if context.target not in context.data:
        raise ValueError(f"Target column is not present: {context.target}.")
    return context.data[context.target]


class ExactTargetCopyCheck:
    """Detect features that exactly reproduce the configured target."""

    check_id = "LEAKAGE-TARGET-001"
    category = "Leakage"

    def run(self, context: AuditContext) -> list[Finding]:
        target = _target_series(context)
        if target is None:
            return []
        findings = []
        for column in context.data.columns:
            if column == context.target:
                continue
            comparable = context.data[column].notna() & target.notna()
            if comparable.any() and context.data.loc[comparable, column].equals(
                target[comparable]
            ):
                findings.append(
                    Finding(
                        check_id=self.check_id,
                        category=self.category,
                        severity=Severity.CRITICAL,
                        title="Feature exactly copies the target",
                        description=(
                            "The feature values exactly match the configured target "
                            "where both values are present."
                        ),
                        column=str(column),
                        evidence={
                            "matching_rows": int(comparable.sum()),
                            "target_column": context.target,
                        },
                        recommendation=(
                            "Trace feature creation and exclude the feature unless "
                            "its availability is guaranteed before prediction."
                        ),
                    )
                )
        return findings


class NumericalAssociationCheck:
    """Screen numeric features for strong Pearson/Spearman association."""

    check_id = "LEAKAGE-NUMERIC-001"
    category = "Leakage"

    def __init__(self, threshold: float = 0.95) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be greater than 0 and at most 1.")
        self.threshold = threshold

    def run(self, context: AuditContext) -> list[Finding]:
        target = _target_series(context)
        if target is None or not pd.api.types.is_numeric_dtype(target):
            return []
        findings = []
        for column in context.data.select_dtypes(include="number").columns:
            if column == context.target:
                continue
            pair = context.data[[column, context.target]].dropna()
            if len(pair) < 3 or pair[column].nunique() < 2:
                continue
            pearson = float(pair[column].corr(pair[context.target], method="pearson"))
            spearman = float(pair[column].corr(pair[context.target], method="spearman"))
            if max(abs(pearson), abs(spearman)) < self.threshold:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=Severity.WARNING,
                    title="Strong numerical target association",
                    description=(
                        "A strong association was detected; this is a screening "
                        "signal, not proof of leakage."
                    ),
                    column=str(column),
                    evidence={
                        "target_column": context.target,
                        "pearson": pearson,
                        "spearman": spearman,
                        "threshold": self.threshold,
                        "sample_size": len(pair),
                    },
                    recommendation=(
                        "Review feature timing and construction; correlation alone "
                        "does not establish leakage or causation."
                    ),
                )
            )
        return findings


class MutualInformationCheck:
    """Screen features using target mutual information."""

    check_id = "LEAKAGE-MUTUAL-INFO-001"
    category = "Leakage"

    def __init__(self, threshold: float = 0.5, random_state: int = 0) -> None:
        if threshold < 0:
            raise ValueError("threshold must not be negative.")
        self.threshold = threshold
        self.random_state = random_state

    def run(self, context: AuditContext) -> list[Finding]:
        target = _target_series(context)
        if target is None:
            return []
        if context.task_type not in {"classification", "regression"}:
            raise ValueError(
                "task_type must be 'classification' or 'regression' for "
                "mutual-information screening."
            )
        candidates = [
            column
            for column in context.data.columns
            if column != context.target
            and pd.api.types.is_numeric_dtype(context.data[column])
        ]
        if not candidates:
            return []
        pair = context.data[candidates + [context.target]].dropna()
        if len(pair) < 4 or target.nunique(dropna=True) < 2:
            return []
        features = pair[candidates]
        target_values = pair[context.target]
        if context.task_type == "classification":
            scores = mutual_info_classif(
                features, target_values, random_state=self.random_state
            )
        elif context.task_type == "regression":
            scores = mutual_info_regression(
                features, target_values, random_state=self.random_state
            )
        findings = []
        for column, score in zip(candidates, scores, strict=True):
            score = float(score)
            if score < self.threshold:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=Severity.WARNING,
                    title="High mutual information with target",
                    description=(
                        "A high mutual-information score was detected; this is a "
                        "screening signal, not proof of leakage."
                    ),
                    column=str(column),
                    evidence={
                        "target_column": context.target,
                        "mutual_information": score,
                        "threshold": self.threshold,
                        "sample_size": len(pair),
                    },
                    recommendation=(
                        "Inspect feature availability and transformations before "
                        "treating this association as usable signal."
                    ),
                )
            )
        return findings


class CategoricalAssociationCheck:
    """Screen categorical features using Cramer's V."""

    check_id = "LEAKAGE-CATEGORICAL-001"
    category = "Leakage"

    def __init__(self, threshold: float = 0.9) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be greater than 0 and at most 1.")
        self.threshold = threshold

    def run(self, context: AuditContext) -> list[Finding]:
        target = _target_series(context)
        if target is None or not (
            pd.api.types.is_object_dtype(target.dtype)
            or pd.api.types.is_string_dtype(target.dtype)
            or isinstance(target.dtype, pd.CategoricalDtype)
        ):
            return []
        findings = []
        for column in context.data.select_dtypes(
            include=["object", "string", "category"]
        ).columns:
            if column == context.target:
                continue
            pair = context.data[[column, context.target]].dropna()
            if len(pair) < 3 or pair[column].nunique() < 2:
                continue
            score = _cramers_v(pair[column], pair[context.target])
            if score < self.threshold:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=Severity.WARNING,
                    title="Strong categorical target association",
                    description=(
                        "A strong categorical association was detected; this is "
                        "a screening signal, not proof of leakage."
                    ),
                    column=str(column),
                    evidence={
                        "target_column": context.target,
                        "cramers_v": score,
                        "threshold": self.threshold,
                        "sample_size": len(pair),
                    },
                    recommendation=(
                        "Review feature timing and category definitions; association "
                        "alone does not establish leakage."
                    ),
                )
            )
        return findings


def _cramers_v(left: Iterable[object], right: Iterable[object]) -> float:
    table = pd.crosstab(pd.Series(left), pd.Series(right))
    observed = table.to_numpy()
    total = observed.sum()
    if total == 0:
        return 0.0
    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / total
    chi_square = ((observed - expected) ** 2 / expected).sum()
    phi_squared = chi_square / total
    rows, columns = observed.shape
    denominator = min(rows - 1, columns - 1)
    return sqrt(phi_squared / denominator) if denominator > 0 else 0.0
