import numpy as np
import pandas as pd
import pytest

from datasleuth.checks import DistributionShiftCheck, ImbalanceCheck
from datasleuth.engine import AuditContext, Severity
from datasleuth.reporting import (
    categorical_distribution_figure,
    numerical_distribution_figure,
)


def test_classification_imbalance_reports_counts_and_ratio():
    data = pd.DataFrame({"target": ["a"] * 9 + ["b"]})

    findings = ImbalanceCheck().run(
        AuditContext(data=data, target="target", task_type="classification")
    )

    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["class_counts"] == {"a": 9, "b": 1}
    assert findings[0].evidence["majority_to_minority_ratio"] == 9.0


def test_regression_target_is_summarized_not_called_class_imbalance():
    data = pd.DataFrame({"target": [1.0, 2.0, 4.0]})

    findings = ImbalanceCheck().run(
        AuditContext(data=data, target="target", task_type="regression")
    )

    assert findings[0].severity == Severity.INFO
    assert findings[0].evidence["median"] == 2.0


def test_numerical_shift_uses_reference_bins_and_ks():
    train = pd.DataFrame({"value": np.arange(20, dtype=float)})
    test = pd.DataFrame({"value": np.arange(100, 120, dtype=float)})

    findings = DistributionShiftCheck(min_sample_size=5).run(
        AuditContext(data=train, train_data=train, test_data=test)
    )

    assert len(findings) == 1
    assert findings[0].column == "value"
    assert findings[0].evidence["reference_dataset"] == "train_data"
    assert findings[0].evidence["psi"] > 0.2
    assert findings[0].evidence["ks_pvalue"] < 0.05


def test_equivalent_numerical_distribution_does_not_trigger():
    train = pd.DataFrame({"value": np.arange(20, dtype=float)})
    test = pd.DataFrame({"value": np.arange(20, dtype=float)})

    assert DistributionShiftCheck(min_sample_size=5).run(
        AuditContext(data=train, train_data=train, test_data=test)
    ) == []


def test_categorical_shift_reports_unseen_categories():
    train = pd.DataFrame({"status": ["A", "A", "B", "B", "A", "B"]})
    test = pd.DataFrame({"status": ["A", "B", "C", "C", "C", "C"]})

    findings = DistributionShiftCheck(min_sample_size=5).run(
        AuditContext(data=train, train_data=train, test_data=test)
    )

    assert len(findings) == 1
    assert findings[0].evidence["unseen_categories"] == ["C"]
    assert findings[0].evidence["psi"] > 0


def test_small_samples_and_missing_split_are_transparent():
    train = pd.DataFrame({"value": [1.0, 2.0]})
    test = pd.DataFrame({"value": [10.0, 20.0]})

    assert DistributionShiftCheck(min_sample_size=5).run(
        AuditContext(data=train, train_data=train, test_data=test)
    ) == []
    assert DistributionShiftCheck().run(AuditContext(data=train)) == []


def test_distribution_configuration_and_target_inputs_are_validated():
    with pytest.raises(ValueError):
        DistributionShiftCheck(bins=1)
    with pytest.raises(ValueError):
        ImbalanceCheck(minority_ratio_threshold=0)
    with pytest.raises(ValueError, match="task_type"):
        ImbalanceCheck().run(
            AuditContext(
                data=pd.DataFrame({"target": [1, 2]}),
                target="target",
            )
        )


def test_plotly_visualizations_return_figures():
    reference = pd.Series([1, 2, 3])
    comparison = pd.Series([2, 3, 4])

    numerical = numerical_distribution_figure(
        reference, comparison, column="value"
    )
    categorical = categorical_distribution_figure(
        pd.Series(["A", "B"]), pd.Series(["B", "C"]), column="status"
    )

    assert len(numerical.data) > 0
    assert len(categorical.data) > 0
