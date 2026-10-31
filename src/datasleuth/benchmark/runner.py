"""Benchmark execution and measured quality metrics."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from datasleuth.checks import (
    CategoryConsistencyCheck,
    ConstantColumnCheck,
    ExactDuplicateCheck,
    ExactTargetCopyCheck,
    MissingValueCheck,
    OutlierCheck,
)
from datasleuth.engine import AuditContext, AuditOrchestrator, CheckRegistry

from .synthetic import GroundTruth, SyntheticDataset

_CHECK_LABELS = {
    "QUALITY-MISSING-001": "missing",
    "QUALITY-DUPLICATES-001": "exact_duplicate",
    "QUALITY-CONSTANT-001": "constant",
    "QUALITY-OUTLIERS-001": "outlier",
    "QUALITY-CATEGORIES-001": "category_inconsistency",
    "LEAKAGE-TARGET-001": "target_copy",
}


@dataclass(frozen=True)
class ClassificationMetrics:
    """Precision, recall, and F1 for one defect label."""

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured benchmark output, including predictions and runtime."""

    metrics: dict[str, ClassificationMetrics]
    predicted: tuple[GroundTruth, ...]
    ground_truth: tuple[GroundTruth, ...]
    runtime_seconds: float
    finding_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible benchmark data."""
        return {
            "metrics": {
                label: asdict(metrics) for label, metrics in self.metrics.items()
            },
            "predicted": [asdict(item) for item in self.predicted],
            "ground_truth": [asdict(item) for item in self.ground_truth],
            "runtime_seconds": self.runtime_seconds,
            "finding_count": self.finding_count,
        }


def run_benchmark(dataset: SyntheticDataset) -> BenchmarkResult:
    """Run real checks against synthetic data and calculate measured metrics."""
    checks = [
        MissingValueCheck(),
        ExactDuplicateCheck(),
        ConstantColumnCheck(),
        OutlierCheck(method="iqr"),
        CategoryConsistencyCheck(),
        ExactTargetCopyCheck(),
    ]
    context = AuditContext(
        data=dataset.data,
        target=dataset.target,
        task_type="classification",
    )
    started = time.perf_counter()
    result = AuditOrchestrator(CheckRegistry(checks)).run(context)
    runtime = time.perf_counter() - started
    predicted = tuple(
        GroundTruth(_CHECK_LABELS[finding.check_id], finding.column)
        for finding in result.findings
        if finding.check_id in _CHECK_LABELS
    )
    return BenchmarkResult(
        metrics=_calculate_metrics(dataset.ground_truth, predicted),
        predicted=predicted,
        ground_truth=dataset.ground_truth,
        runtime_seconds=runtime,
        finding_count=len(result.findings),
    )


def _calculate_metrics(
    expected: tuple[GroundTruth, ...],
    predicted: tuple[GroundTruth, ...],
) -> dict[str, ClassificationMetrics]:
    labels = sorted(
        {item.label for item in expected} | {item.label for item in predicted}
    )
    expected_set = {(item.label, item.column) for item in expected}
    predicted_set = {(item.label, item.column) for item in predicted}
    metrics = {}
    for label in labels:
        expected_label = {item for item in expected_set if item[0] == label}
        predicted_label = {item for item in predicted_set if item[0] == label}
        true_positive = len(expected_label & predicted_label)
        false_positive = len(predicted_label - expected_label)
        false_negative = len(expected_label - predicted_label)
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2 * precision * recall, precision + recall)
        metrics[label] = ClassificationMetrics(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=precision,
            recall=recall,
            f1=f1,
        )
    return metrics


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
