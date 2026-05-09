"""
core/logging.py — Centralized logging configuration.

All modules call get_logger(__name__) to get their logger.
configure_logging() is called once at startup from cli/main.py.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

_LOG_FILE = "investment_bot.log"
_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_file: str | None = _LOG_FILE,
    json_mode: bool = False,
) -> None:
    """
    Configure root logger once at startup.

    json_mode: emit JSON lines instead of human-readable text.
    Set DAILY_INVESTOR_LOG_LEVEL env var to override level at runtime.
    """
    env_level = os.environ.get("DAILY_INVESTOR_LOG_LEVEL", level).upper()
    numeric_level = getattr(logging, env_level, logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    if json_mode:
        fmt = _JsonFormatter()
        for h in handlers:
            h.setFormatter(fmt)
    else:
        formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)
        for h in handlers:
            h.setFormatter(formatter)

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Use as: logger = get_logger(__name__)"""
    return logging.getLogger(name)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON-lines formatter for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback

        payload: dict = {
            "ts": self.formatTime(record, self.datefmt or _DATE_FMT),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)
