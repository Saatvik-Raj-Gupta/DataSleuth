"""Logging setup for command-line and application entry points."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a concise package-wide logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

