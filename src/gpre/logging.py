"""structlog setup: pretty console output plus one JSON-lines file per run in ``results/logs/``."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, TextIO

import structlog

_file_handle: TextIO | None = None


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure structlog for the process.

    Args:
        level: minimum level (DEBUG, INFO, WARNING, ERROR).
        log_file: if given, every event is also written there as one JSON object per line.
    """
    global _file_handle
    numeric = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    console = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    def _render(logger: Any, name: str, event: MutableMapping[str, Any]) -> str:
        if _file_handle is not None:
            _file_handle.write(str(structlog.processors.JSONRenderer()(logger, name, dict(event))))
            _file_handle.write("\n")
            _file_handle.flush()
        return str(console(logger, name, event))

    if _file_handle is not None:
        _file_handle.close()
        _file_handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _file_handle = log_file.open("a", encoding="utf-8")

    processors: list[Any] = [*shared, _render]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        # Resolve sys.stderr per logger so a swapped stream (tests, CLI runners) is honoured.
        logger_factory=lambda *_args: structlog.PrintLogger(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> Any:
    """Module logger."""
    return structlog.get_logger(name)
