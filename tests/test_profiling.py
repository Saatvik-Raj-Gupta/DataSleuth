import pandas as pd

from datasleuth.profiling import profile_dataframe


def test_profile_dataframe_reports_dataset_and_column_statistics():
    frame = pd.DataFrame(
        {
            "number": [1, 2, None],
            "category": ["A", "A", "B"],
            "flag": [True, False, True],
            "when": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "constant": ["same", "same", "same"],
        }
    )
    original = frame.copy(deep=True)

    result = profile_dataframe(frame)
    by_name = {profile.name: profile for profile in result.column_profiles}

    assert result.row_count == 3
    assert result.column_count == 5
    assert result.memory_bytes > 0
    assert by_name["number"].kind == "numeric"
    assert by_name["number"].missing_count == 1
    assert by_name["number"].missing_percentage == 33.3333
    assert by_name["number"].numeric_statistics == {
        "count": 2,
        "mean": 1.5,
        "std": 0.7071067811865476,
        "min": 1.0,
        "median": 1.5,
        "max": 2.0,
    }
    assert by_name["category"].kind == "categorical"
    assert by_name["category"].categorical_frequencies == {"A": 2, "B": 1}
    assert by_name["flag"].kind == "boolean"
    assert by_name["when"].kind == "datetime"
    assert by_name["constant"].is_constant is True
    pd.testing.assert_frame_equal(frame, original)


def test_profile_handles_all_null_and_empty_dataframes():
    frame = pd.DataFrame({"empty": [None, None], "number": [1.0, 2.0]})
    empty_frame = pd.DataFrame({"number": pd.Series(dtype="float64")})

    result = profile_dataframe(frame)
    by_name = {profile.name: profile for profile in result.column_profiles}
    empty_result = profile_dataframe(empty_frame)
    empty_profile = empty_result.column_profiles[0]

    assert result.row_count == 2
    assert by_name["empty"].is_constant is True
    assert by_name["empty"].unique_count == 0
    assert by_name["empty"].categorical_frequencies == {"<MISSING>": 2}
    assert empty_profile.row_count == 0
    assert empty_profile.numeric_statistics == {
        "count": 0,
        "mean": 0,
        "std": 0,
        "min": 0,
        "median": 0,
        "max": 0,
    }


def test_datetime_inference_is_opt_in_and_conservative():
    frame = pd.DataFrame(
        {
            "date_text": ["2026-01-01", "2026-01-02"],
            "mixed_text": ["2026-01-01", "not a date"],
        }
    )

    without_inference = profile_dataframe(frame)
    with_inference = profile_dataframe(frame, infer_datetime=True)

    assert without_inference.column_profiles[0].kind == "categorical"
    assert with_inference.column_profiles[0].kind == "datetime"
    assert with_inference.column_profiles[0].datetime_inferred is True
    assert with_inference.column_profiles[1].kind == "categorical"


def test_profile_serializes_to_nested_dictionary():
    result = profile_dataframe(pd.DataFrame({"value": [1, 2]}))

    payload = result.to_dict()

    assert payload["row_count"] == 2
    assert payload["column_profiles"][0]["numeric_statistics"]["mean"] == 1.5
