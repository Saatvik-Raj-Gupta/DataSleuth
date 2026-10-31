"""Errors raised while validating or loading input datasets."""


class DataSleuthError(Exception):
    """Base class for expected DataSleuth errors."""


class IngestionError(DataSleuthError):
    """An uploaded dataset cannot be safely loaded."""

