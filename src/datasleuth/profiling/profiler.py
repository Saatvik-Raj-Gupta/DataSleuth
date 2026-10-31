"""Deterministic, non-mutating profiling for tabular data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

ColumnKind = Literal["numeric", "categorical", "boolean", "datetime", "unknown"]


@dataclass(frozen=True)
class ColumnProfile:
    """Summary statistics for one input column."""

    name: str
    dtype: str
    kind: ColumnKind
    row_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_constant: bool
    memory_bytes: int
    numeric_statistics: Mapping[str, float | int] | None = None
    categorical_frequencies: Mapping[str, int] | None = None
    datetime_statistics: Mapping[str, str] | None = None
    datetime_inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class DatasetProfile:
    """Summary statistics for a complete input DataFrame."""

    row_count: int
    column_count: int
    memory_bytes: int
    column_profiles: tuple[ColumnProfile, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


def profile_dataframe(
    frame: pd.DataFrame,
    *,
    infer_datetime: bool = False,
) -> DatasetProfile:
    """Profile a DataFrame without modifying it.

    Datetime inference is disabled by default. When enabled, object/string
    columns are inferred only when every non-null value parses successfully.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    column_profiles = tuple(
        _profile_column(name, series, infer_datetime=infer_datetime)
        for name, series in frame.items()
    )
    return DatasetProfile(
        row_count=len(frame),
        column_count=len(frame.columns),
        memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
        column_profiles=column_profiles,
    )


def _profile_column(
    name: Any,
    series: pd.Series,
    *,
    infer_datetime: bool,
) -> ColumnProfile:
    missing_count = int(series.isna().sum())
    row_count = len(series)
    unique_count = int(series.nunique(dropna=True))
    non_missing = row_count - missing_count
    kind, datetime_values, datetime_inferred = _classify(
        series, infer_datetime=infer_datetime
    )

    numeric_statistics = (
        _numeric_statistics(series) if kind == "numeric" else None
    )
    categorical_frequencies = (
        _categorical_frequencies(series) if kind == "categorical" else None
    )
    datetime_statistics = (
        _datetime_statistics(datetime_values) if kind == "datetime" else None
    )

    return ColumnProfile(
        name=str(name),
        dtype=str(series.dtype),
        kind=kind,
        row_count=row_count,
        missing_count=missing_count,
        missing_percentage=_percentage(missing_count, row_count),
        unique_count=unique_count,
        unique_percentage=_percentage(unique_count, non_missing),
        is_constant=unique_count <= 1,
        memory_bytes=int(series.memory_usage(index=False, deep=True)),
        numeric_statistics=numeric_statistics,
        categorical_frequencies=categorical_frequencies,
        datetime_statistics=datetime_statistics,
        datetime_inferred=datetime_inferred,
    )


def _classify(
    series: pd.Series,
    *,
    infer_datetime: bool,
) -> tuple[ColumnKind, pd.Series | None, bool]:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean", None, False
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric", None, False
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime", series, False
    if infer_datetime and series.notna().any():
        converted = pd.to_datetime(series, errors="coerce")
        non_missing = series.notna()
        if converted[non_missing].notna().all():
            return "datetime", converted, True
    if (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or pd.api.types.is_object_dtype(series.dtype)
    ):
        return "categorical", None, False
    return "unknown", None, False


def _numeric_statistics(series: pd.Series) -> dict[str, float | int]:
    summary = series.describe()
    return {
        "count": _number(summary.get("count", 0)),
        "mean": _number(summary.get("mean")),
        "std": _number(summary.get("std")),
        "min": _number(summary.get("min")),
        "median": _number(series.median()),
        "max": _number(summary.get("max")),
    }


def _categorical_frequencies(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {_display_value(value): int(count) for value, count in counts.items()}


def _datetime_statistics(series: pd.Series | None) -> dict[str, str] | None:
    if series is None or series.dropna().empty:
        return None
    return {
        "min": series.min().isoformat(),
        "max": series.max().isoformat(),
    }


def _display_value(value: Any) -> str:
    return "<MISSING>" if pd.isna(value) else str(value)


def _number(value: Any) -> float | int:
    if pd.isna(value):
        return 0
    converted = value.item() if hasattr(value, "item") else value
    return int(converted) if isinstance(converted, int) else float(converted)


def _percentage(numerator: int, denominator: int) -> float:
    return round(numerator * 100 / denominator, 4) if denominator else 0.0
