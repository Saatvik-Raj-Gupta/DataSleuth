import pandas as pd
import pytest

from datasleuth.checks import (
    CategoricalAssociationCheck,
    ExactTargetCopyCheck,
    IdLikeCheck,
    MutualInformationCheck,
    NumericalAssociationCheck,
    TemporalContextCheck,
    TrainTestOverlapCheck,
)
from datasleuth.engine import AuditContext, Severity


def test_exact_target_copy_is_critical():
    data = pd.DataFrame(
        {"target": [0, 1, 0, 1], "copied_target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]}
    )

    findings = ExactTargetCopyCheck().run(
        AuditContext(data=data, target="target", task_type="classification")
    )

    assert len(findings) == 1
    assert findings[0].column == "copied_target"
    assert findings[0].severity == Severity.CRITICAL


def test_numerical_and_categorical_association_are_warnings_not_confirmed_leakage():
    numeric = pd.DataFrame(
        {"target": [0, 1, 0, 1, 0, 1], "feature": [0, 2, 0, 2, 0, 2]}
    )
    categorical = pd.DataFrame(
        {
            "target": ["no", "yes", "no", "yes", "no", "yes"],
            "feature": ["N", "Y", "N", "Y", "N", "Y"],
        }
    )

    numerical = NumericalAssociationCheck().run(
        AuditContext(data=numeric, target="target", task_type="classification")
    )
    categorical_findings = CategoricalAssociationCheck().run(
        AuditContext(data=categorical, target="target", task_type="classification")
    )

    assert numerical[0].severity == Severity.WARNING
    assert "not proof of leakage" in numerical[0].description
    assert categorical_findings[0].evidence["cramers_v"] > 0.9


def test_mutual_information_detects_injected_classification_signal():
    data = pd.DataFrame(
        {
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
            "feature": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    findings = MutualInformationCheck(threshold=0.1).run(
        AuditContext(data=data, target="target", task_type="classification")
    )

    assert findings[0].column == "feature"
    assert findings[0].evidence["mutual_information"] >= 0.1


def test_id_like_and_temporal_checks_are_explicit_heuristics():
    data = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=4),
            "record_id": ["a", "b", "c", "d"],
        }
    )
    context = AuditContext(data=data)

    id_findings = IdLikeCheck().run(context)
    temporal_findings = TemporalContextCheck("event_time").run(context)

    assert id_findings[0].severity == Severity.INFO
    assert "not leakage" in id_findings[0].recommendation
    assert temporal_findings[0].column == "event_time"


def test_train_test_overlap_is_critical():
    train = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    test = pd.DataFrame({"id": [2, 3], "value": ["b", "c"]})

    findings = TrainTestOverlapCheck().run(
        AuditContext(data=train, train_data=train, test_data=test)
    )

    assert findings[0].severity == Severity.CRITICAL
    assert findings[0].evidence["overlap_rows"] == 1


def test_small_constant_or_missing_targets_do_not_crash():
    small = pd.DataFrame({"target": [1, 1], "feature": [1, 2]})
    missing = pd.DataFrame({"target": [None, None, None], "feature": [1, 2, 3]})

    assert NumericalAssociationCheck().run(
        AuditContext(data=small, target="target", task_type="regression")
    ) == []
    assert MutualInformationCheck().run(
        AuditContext(data=missing, target="target", task_type="regression")
    ) == []
    assert ExactTargetCopyCheck().run(
        AuditContext(data=missing, target="target")
    ) == []


def test_leakage_checks_validate_configuration_and_inputs():
    data = pd.DataFrame({"target": [0, 1], "feature": [1, 2]})

    with pytest.raises(ValueError, match="Target column"):
        ExactTargetCopyCheck().run(AuditContext(data=data, target="missing"))
    with pytest.raises(ValueError, match="task_type"):
        MutualInformationCheck().run(AuditContext(data=data, target="target"))
    with pytest.raises(ValueError, match="no common"):
        TrainTestOverlapCheck().run(
            AuditContext(
                data=data,
                train_data=data,
                test_data=pd.DataFrame({"other": [1, 2]}),
            )
        )
