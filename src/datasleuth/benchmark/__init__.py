"""Deterministic benchmark and hardening utilities."""

from .runner import BenchmarkResult, run_benchmark
from .synthetic import SyntheticDataset, generate_synthetic_dataset

__all__ = [
    "BenchmarkResult",
    "SyntheticDataset",
    "generate_synthetic_dataset",
    "run_benchmark",
]

