from io import BytesIO

import pandas as pd
import pytest

from datasleuth.ingestion import CsvLoadConfig, load_csv
from datasleuth.ingestion.errors import IngestionError


def test_load_csv_preserves_values_and_columns(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\nAda,3\nGrace,4\n", encoding="utf-8")

    result = load_csv(path)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"name": ["Ada", "Grace"], "value": [3, 4]}),
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"header\n", "at least one data row"),
    ],
)
def test_load_csv_rejects_empty_inputs(content, message):
    with pytest.raises(IngestionError, match=message):
        load_csv(BytesIO(content))


def test_load_csv_rejects_malformed_content():
    with pytest.raises(IngestionError, match="Could not parse CSV"):
        load_csv(BytesIO(b"a,b\n\"unterminated,2\n"))


def test_load_csv_rejects_invalid_encoding():
    with pytest.raises(IngestionError, match="Could not decode CSV"):
        load_csv(BytesIO(b"name\ncaf\xe9\n"))


def test_load_csv_rejects_non_csv_path(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(IngestionError, match="Only .csv"):
        load_csv(path)


def test_load_csv_enforces_size_limit():
    with pytest.raises(IngestionError, match="size limit"):
        load_csv(BytesIO(b"a\n1\n"), config=CsvLoadConfig(max_bytes=2))
