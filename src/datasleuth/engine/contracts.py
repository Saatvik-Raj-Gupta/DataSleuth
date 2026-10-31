"""Typed contracts shared by audit checks and reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import pandas as pd


class Severity(StrEnum):
    """Finding severity ordered by impact."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class Finding:
    """A serializable, evidence-backed audit finding."""

    check_id: str
    category: str
    severity: Severity
    title: str
    description: str
    column: str | None
    evidence: Mapping[str, Any]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible finding data."""
        return json.loads(self.to_json())

    def to_json(self) -> str:
        """Serialize this finding as JSON."""
        return json.dumps(asdict(self), default=_json_default, sort_keys=True)


@dataclass(frozen=True)
class AuditContext:
    """Immutable configuration wrapper passed to independent checks."""

    data: pd.DataFrame
    target: str | None = None
    task_type: TaskType | None = None
    train_data: pd.DataFrame | None = None
    test_data: pd.DataFrame | None = None
    config: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")
