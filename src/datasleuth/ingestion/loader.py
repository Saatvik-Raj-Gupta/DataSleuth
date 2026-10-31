"""Safe CSV loading without modifying the uploaded data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd

from .errors import IngestionError


@dataclass(frozen=True)
class CsvLoadConfig:
    """Limits and parser options applied to CSV uploads."""

    max_bytes: int = 100 * 1024 * 1024
    encoding: str = "utf-8"
    separator: str = ","


def load_csv(
    source: str | Path | IO[bytes] | IO[str],
    *,
    config: CsvLoadConfig | None = None,
) -> pd.DataFrame:
    """Load a non-empty CSV after validating its size and parseability.

    The returned DataFrame is newly created by pandas; the source is never
    modified. Parsing errors are converted to a stable, user-facing error type.
    """
    settings = config or CsvLoadConfig()
    _validate_config(settings)
    _validate_source(source, settings)

    try:
        frame = pd.read_csv(
            source,
            encoding=settings.encoding,
            sep=settings.separator,
            on_bad_lines="error",
        )
    except (UnicodeDecodeError, LookupError) as exc:
        raise IngestionError(
            f"Could not decode CSV using encoding {settings.encoding!r}."
        ) from exc
    except (pd.errors.ParserError, ValueError, OSError) as exc:
        raise IngestionError(f"Could not parse CSV: {exc}") from exc

    if frame.empty:
        raise IngestionError("CSV must contain at least one data row.")
    if frame.shape[1] == 0:
        raise IngestionError("CSV must contain at least one column.")
    return frame


def _validate_config(config: CsvLoadConfig) -> None:
    if config.max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero.")
    if not config.separator or len(config.separator) != 1:
        raise ValueError("separator must be exactly one character.")


def _validate_source(
    source: str | Path | IO[bytes] | IO[str], config: CsvLoadConfig
) -> None:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() != ".csv":
            raise IngestionError("Only .csv files are supported.")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise IngestionError(f"Could not access CSV file: {exc}") from exc
        if size == 0:
            raise IngestionError("CSV file is empty.")
        if size > config.max_bytes:
            raise IngestionError(
                f"CSV file exceeds the {config.max_bytes} byte size limit."
            )
        return

    if hasattr(source, "seek") and hasattr(source, "tell"):
        try:
            current = source.tell()
            source.seek(0, 2)
            size = source.tell()
            source.seek(current)
        except (OSError, ValueError) as exc:
            raise IngestionError(f"Could not inspect CSV input: {exc}") from exc
        if size == 0:
            raise IngestionError("CSV input is empty.")
        if size > config.max_bytes:
            raise IngestionError(
                f"CSV input exceeds the {config.max_bytes} byte size limit."
            )

