import pandas as pd
import pytest

from datasleuth.benchmark import generate_synthetic_dataset, run_benchmark
from datasleuth.ingestion import CsvLoadConfig, load_csv
from datasleuth.ingestion.errors import IngestionError


def test_synthetic_generation_is_deterministic_and_truth_is_separate():
    first = generate_synthetic_dataset(seed=7, rows=40)
    second = generate_synthetic_dataset(seed=7, rows=40)

    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.ground_truth == second.ground_truth
    assert first.ground_truth
    assert first.ground_truth is not second.ground_truth


def test_clean_synthetic_data_has_no_ground_truth_defects():
    dataset = generate_synthetic_dataset(seed=7, rows=40, clean=True)

    assert dataset.ground_truth == ()
    assert "target_copy" not in dataset.data


def test_benchmark_reports_measured_metrics_and_runtime():
    result = run_benchmark(generate_synthetic_dataset(seed=7, rows=100))

    assert result.runtime_seconds > 0
    assert result.finding_count > 0
    assert set(result.metrics) >= {
        "missing",
        "exact_duplicate",
        "constant",
        "outlier",
        "category_inconsistency",
        "target_copy",
    }
    assert result.metrics["target_copy"].recall == 1.0
    assert result.to_dict()["runtime_seconds"] == result.runtime_seconds


def test_clean_data_false_positive_evaluation_is_explicit():
    dataset = generate_synthetic_dataset(seed=3, rows=100, clean=True)
    result = run_benchmark(dataset)

    assert dataset.ground_truth == ()
    assert result.finding_count == 0
    assert all(
        metric.true_positive == 0 and metric.false_negative == 0
        for metric in result.metrics.values()
    )


@pytest.mark.parametrize(
    ("content", "config"),
    [
        (b"a,b\n1,2\n", CsvLoadConfig(max_bytes=3)),
        (b"a,b\n\"broken,2\n", CsvLoadConfig()),
    ],
)
def test_benchmark_hardening_keeps_malformed_and_oversized_inputs_controlled(
    content, config
):
    with pytest.raises(IngestionError):
        load_csv(__import__("io").BytesIO(content), config=config)
