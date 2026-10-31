import importlib.util
from pathlib import Path

_APP_PATH = Path(__file__).parents[1] / "app.py"
_SPEC = importlib.util.spec_from_file_location("datasleuth_app", _APP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_APP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_APP)

build_checks = _APP.build_checks
selected_check_ids = _APP.selected_check_ids


def test_ui_selection_marks_unconfigured_target_and_split_checks_skipped():
    checks = build_checks(
        missing_threshold=0.0,
        near_constant_threshold=0.99,
        outlier_method="iqr",
        numerical_threshold=0.95,
        categorical_threshold=0.9,
        mutual_information_threshold=0.5,
        id_uniqueness_threshold=0.98,
        psi_threshold=0.2,
        key_columns=[],
        allowed_values={},
        event_time_column=None,
    )

    selected = selected_check_ids(checks, has_target=False, has_split=False)

    assert "QUALITY-MISSING-001" in selected
    assert "LEAKAGE-TARGET-001" not in selected
    assert "LEAKAGE-SPLIT-001" not in selected
    assert len(checks) >= 10


def test_ui_selection_includes_target_and_split_checks_when_configured():
    checks = build_checks(
        missing_threshold=0.0,
        near_constant_threshold=0.99,
        outlier_method="both",
        numerical_threshold=0.95,
        categorical_threshold=0.9,
        mutual_information_threshold=0.5,
        id_uniqueness_threshold=0.98,
        psi_threshold=0.2,
        key_columns=["id"],
        allowed_values={},
        event_time_column="event_time",
    )

    selected = selected_check_ids(checks, has_target=True, has_split=True)

    assert "LEAKAGE-TARGET-001" in selected
    assert "LEAKAGE-SPLIT-001" in selected
    assert "QUALITY-DUPLICATES-002" in selected
    assert "LEAKAGE-TEMPORAL-001" in selected
