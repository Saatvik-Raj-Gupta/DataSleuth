"""Class imbalance and train/test distribution checks."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from datasleuth.engine import AuditContext, Finding, Severity


def _split_data(context: AuditContext) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if context.train_data is None or context.test_data is None:
        return None
    return context.train_data, context.test_data


class ImbalanceCheck:
    """Summarize classification imbalance or regression target distribution."""

    check_id = "DISTRIBUTION-IMBALANCE-001"
    category = "Distribution"

    def __init__(self, minority_ratio_threshold: float = 0.1) -> None:
        if not 0 < minority_ratio_threshold <= 1:
            raise ValueError("minority_ratio_threshold must be between 0 and 1.")
        self.minority_ratio_threshold = minority_ratio_threshold

    def run(self, context: AuditContext) -> list[Finding]:
        if context.target is None:
            return []
        if context.target not in context.data:
            raise ValueError(f"Target column is not present: {context.target}.")
        target = context.data[context.target].dropna()
        if target.empty:
            return []
        if context.task_type == "classification":
            counts = target.value_counts()
            percentages = (counts / len(target) * 100).round(4)
            minority_fraction = float(counts.min() / len(target))
            if len(counts) < 2:
                return []
            severity = (
                Severity.WARNING
                if minority_fraction <= self.minority_ratio_threshold
                else Severity.INFO
            )
            return [
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=severity,
                    title="Classification target distribution",
                    description=(
                        f"The minority class represents {minority_fraction:.2%} "
                        "of non-missing target values."
                    ),
                    column=context.target,
                    evidence={
                        "class_counts": {str(k): int(v) for k, v in counts.items()},
                        "class_percentages": {
                            str(k): float(v) for k, v in percentages.items()
                        },
                        "minority_fraction": minority_fraction,
                        "majority_to_minority_ratio": float(
                            counts.max() / counts.min()
                        ),
                        "threshold": self.minority_ratio_threshold,
                    },
                    recommendation=(
                        "Review stratification, evaluation metrics, and class-aware "
                        "training choices before modeling."
                    ),
                )
            ]
        if context.task_type == "regression":
            return [
                Finding(
                    check_id=self.check_id,
                    category=self.category,
                    severity=Severity.INFO,
                    title="Regression target distribution",
                    description="The target distribution is summarized for review.",
                    column=context.target,
                    evidence={
                        "count": len(target),
                        "mean": float(target.mean()),
                        "std": float(target.std()) if len(target) > 1 else 0.0,
                        "min": float(target.min()),
                        "median": float(target.median()),
                        "max": float(target.max()),
                    },
                    recommendation=(
                        "Review target scale and tails when selecting metrics and "
                        "regression transformations."
                    ),
                )
            ]
        raise ValueError(
            "task_type must be 'classification' or 'regression' for imbalance checks."
        )


class DistributionShiftCheck:
    """Compare compatible train/test columns using PSI and optional KS tests."""

    check_id = "DISTRIBUTION-SHIFT-001"
    category = "Distribution"

    def __init__(
        self,
        *,
        psi_threshold: float = 0.2,
        ks_pvalue_threshold: float = 0.05,
        bins: int = 10,
        run_ks: bool = True,
        min_sample_size: int = 5,
    ) -> None:
        if psi_threshold <= 0 or ks_pvalue_threshold <= 0:
            raise ValueError("distribution thresholds must be greater than zero.")
        if bins < 2 or min_sample_size < 2:
            raise ValueError("bins must be at least 2 and min_sample_size at least 2.")
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.bins = bins
        self.run_ks = run_ks
        self.min_sample_size = min_sample_size

    def run(self, context: AuditContext) -> list[Finding]:
        split = _split_data(context)
        if split is None:
            return []
        reference, comparison = split
        columns = list(reference.columns.intersection(comparison.columns))
        findings = []
        for column in columns:
            if pd.api.types.is_numeric_dtype(
                reference[column]
            ) and pd.api.types.is_numeric_dtype(comparison[column]):
                finding = self._numeric_finding(
                    reference[column], comparison[column], column
                )
            elif _is_categorical(reference[column]) and _is_categorical(
                comparison[column]
            ):
                finding = self._categorical_finding(
                    reference[column], comparison[column], column
                )
            else:
                continue
            if finding is not None:
                findings.append(finding)
        return findings

    def _numeric_finding(
        self, reference: pd.Series, comparison: pd.Series, column: str
    ) -> Finding | None:
        reference_values = reference.dropna().to_numpy(dtype=float)
        comparison_values = comparison.dropna().to_numpy(dtype=float)
        if (
            len(reference_values) < self.min_sample_size
            or len(comparison_values) < self.min_sample_size
            or np.all(reference_values == reference_values[0])
        ):
            return None
        edges = _reference_bins(reference_values, self.bins)
        reference_proportions = _bin_proportions(reference_values, edges)
        comparison_proportions = _bin_proportions(comparison_values, edges)
        psi = _psi(reference_proportions, comparison_proportions)
        ks_statistic, ks_pvalue = ks_2samp(reference_values, comparison_values)
        shifted = psi >= self.psi_threshold or (
            self.run_ks and ks_pvalue < self.ks_pvalue_threshold
        )
        if not shifted:
            return None
        return Finding(
            check_id=self.check_id,
            category=self.category,
            severity=Severity.WARNING,
            title="Numerical distribution shift detected",
            description=(
                "The comparison distribution differs from the reference based on "
                "configurable PSI and/or KS heuristics."
            ),
            column=str(column),
            evidence={
                "reference_dataset": "train_data",
                "comparison_dataset": "test_data",
                "psi": psi,
                "psi_threshold": self.psi_threshold,
                "ks_statistic": float(ks_statistic),
                "ks_pvalue": float(ks_pvalue),
                "ks_pvalue_threshold": self.ks_pvalue_threshold,
                "reference_sample_size": len(reference_values),
                "comparison_sample_size": len(comparison_values),
                "bin_edges": edges.tolist(),
            },
            recommendation=(
                "Investigate collection, sampling, and time-window differences; "
                "PSI and KS are distribution heuristics, not causal explanations."
            ),
        )

    def _categorical_finding(
        self, reference: pd.Series, comparison: pd.Series, column: str
    ) -> Finding | None:
        reference_values = reference.dropna()
        comparison_values = comparison.dropna()
        if (
            len(reference_values) < self.min_sample_size
            or len(comparison_values) < self.min_sample_size
        ):
            return None
        categories = sorted(
            {str(value) for value in reference_values}
            | {str(value) for value in comparison_values}
        )
        reference_proportions = _category_proportions(reference_values, categories)
        comparison_proportions = _category_proportions(comparison_values, categories)
        psi = _psi(reference_proportions, comparison_proportions)
        unseen = sorted(
            {str(value) for value in comparison_values}
            - {str(value) for value in reference_values}
        )
        if psi < self.psi_threshold and not unseen:
            return None
        return Finding(
            check_id=self.check_id,
            category=self.category,
            severity=Severity.WARNING,
            title="Categorical distribution shift detected",
            description=(
                "Category proportions differ from the reference or include unseen "
                "comparison categories."
            ),
            column=str(column),
            evidence={
                "reference_dataset": "train_data",
                "comparison_dataset": "test_data",
                "psi": psi,
                "psi_threshold": self.psi_threshold,
                "unseen_categories": unseen,
                "reference_sample_size": len(reference_values),
                "comparison_sample_size": len(comparison_values),
            },
            recommendation=(
                "Investigate category definitions and source populations; handle "
                "unseen categories explicitly before scoring."
            ),
        )


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _reference_bins(values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([values.min() - 0.5, values.max() + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts / len(values)


def _category_proportions(
    values: Iterable[object], categories: list[str]
) -> np.ndarray:
    counts = pd.Series([str(value) for value in values]).value_counts()
    return np.array([counts.get(category, 0) for category in categories]) / len(values)


def _psi(reference: np.ndarray, comparison: np.ndarray) -> float:
    epsilon = 1e-6
    reference = np.clip(reference, epsilon, None)
    comparison = np.clip(comparison, epsilon, None)
    return float(np.sum((comparison - reference) * np.log(comparison / reference)))
