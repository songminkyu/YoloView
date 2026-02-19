"""GLM-OCR Logging utilities.

Provides a unified logging system with:
- INFO level: User-facing progress messages
- DEBUG level: Detailed debugging info + profiling

Usage:
    from glmocr.utils.logging import get_logger, profile

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.debug("Detailed info: %s", data)

    # Profiling (only logs in DEBUG mode)
    with profile("image_encoding"):
        encode_image(...)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

# Package-level logger
_PACKAGE_LOGGER_NAME = "glmocr"
_configured = False
_configured_source: Optional[str] = None  # "auto" | "explicit" | None
# Default format
_INFO_FORMAT = "%(message)s"
_DEBUG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"


class ProfileLogger:
    """Profiling logger for measuring execution time.

    Only logs when DEBUG level is enabled.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled (DEBUG level)."""
        return self._logger.isEnabledFor(logging.DEBUG)

    def log(self, label: str, elapsed_ms: float) -> None:
        """Log a profiling result."""
        if self.enabled:
            self._logger.debug("[profile] %s: %.1fms", label, elapsed_ms)

    @contextmanager
    def measure(self, label: str):
        """Context manager for measuring execution time.

        Example:
            with profiler.measure("image_encoding"):
                encode_image(...)
        """
        if not self.enabled:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.log(label, elapsed_ms)

    def __call__(self, label: str):
        """Decorator for measuring function execution time.

        Example:
            @profiler("process_page")
            def process_page(...):
                ...
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                t0 = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self.log(label, elapsed_ms)

            return wrapper

        return decorator


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    *,
    _source: str = "explicit",
) -> None:
    """Configure the glmocr package logging.

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR").
        format_string: Custom format string. If None, uses default based on level.
    """
    global _configured, _configured_source

    # Get or create package logger
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)

    # Parse level
    level_value = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level_value)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(level_value)

    # Set format
    if format_string is None:
        format_string = _DEBUG_FORMAT if level_value == logging.DEBUG else _INFO_FORMAT

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _configured = True
    _configured_source = _source


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    global _configured

    # Ensure package logging is configured
    if not _configured:
        configure_logging(_source="auto")

    # Return child logger under package namespace
    if name.startswith(_PACKAGE_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{name}")


def get_profiler(name: str) -> ProfileLogger:
    """Get a profiler for the given module.

    Args:
        name: Module name (typically __name__).

    Returns:
        ProfileLogger instance.

    Example:
        profiler = get_profiler(__name__)

        with profiler.measure("encoding"):
            encode()

        # Or as decorator
        @profiler("process")
        def process():
            ...
    """
    logger = get_logger(name)
    return ProfileLogger(logger)


def set_log_level(level: str) -> None:
    """Set the log level for the glmocr package.

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    configure_logging(level=level)


def ensure_logging_configured(
    level: str = "INFO", format_string: Optional[str] = None
) -> None:
    """Ensure glmocr logging is configured from an external config.

    If logging was only auto-configured (the implicit default triggered by get_logger()),
    this will reconfigure it using the provided level/format.

    This avoids the common pitfall where importing modules creates the logger and locks
    logging into the default INFO level even when a config file specifies DEBUG.
    """
    global _configured, _configured_source

    if (not _configured) or (_configured_source == "auto"):
        configure_logging(level=level, format_string=format_string, _source="explicit")
