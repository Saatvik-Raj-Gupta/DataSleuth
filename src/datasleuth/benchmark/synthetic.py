"""Deterministic synthetic datasets with separate defect ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

DefectLabel = Literal[
    "missing",
    "exact_duplicate",
    "constant",
    "outlier",
    "category_inconsistency",
    "target_copy",
]


@dataclass(frozen=True)
class GroundTruth:
    """One injected defect and its affected column, if applicable."""

    label: DefectLabel
    column: str | None


@dataclass(frozen=True)
class SyntheticDataset:
    """Synthetic data and independently stored injected-defect truth."""

    data: pd.DataFrame
    target: str
    ground_truth: tuple[GroundTruth, ...]


def generate_synthetic_dataset(
    *,
    seed: int = 0,
    rows: int = 200,
    clean: bool = False,
) -> SyntheticDataset:
    """Generate a deterministic tabular dataset for benchmark evaluation."""
    if rows < 20:
        raise ValueError("rows must be at least 20.")
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=rows)
    data = pd.DataFrame(
        {
            "numeric": rng.normal(50, 10, size=rows),
            "category": np.where(target == 1, "A", "B"),
            "stable_group": np.where(np.arange(rows) % 2, "left", "right"),
            "target": target,
        }
    )
    if clean:
        return SyntheticDataset(data=data, target="target", ground_truth=())

    ground_truth: list[GroundTruth] = []
    missing_index = np.arange(0, rows, max(1, rows // 20))
    data.loc[missing_index, "numeric"] = np.nan
    ground_truth.append(GroundTruth("missing", "numeric"))

    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    ground_truth.append(GroundTruth("exact_duplicate", None))

    data["constant"] = "constant"
    ground_truth.append(GroundTruth("constant", "constant"))

    outlier_index = len(data) - 1
    data.loc[outlier_index, "numeric"] = 1000.0
    ground_truth.append(GroundTruth("outlier", "numeric"))

    data.loc[0, "category"] = " a "
    ground_truth.append(GroundTruth("category_inconsistency", "category"))

    data["target_copy"] = data["target"]
    ground_truth.append(GroundTruth("target_copy", "target_copy"))
    return SyntheticDataset(
        data=data,
        target="target",
        ground_truth=tuple(ground_truth),
    )
