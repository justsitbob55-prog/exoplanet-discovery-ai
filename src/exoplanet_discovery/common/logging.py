"""Centralized logging configuration for the exoplanet discovery package.

Provides structured logging with sensible defaults. Uses Python's built-in
logging module without external dependencies.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

# Global logger instance
_logger: logging.Logger | None = None


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure application-wide logging with console and optional file handlers.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to log file. If provided, logs are written to file.
        format_string: Custom log format. If None, uses default structured format.

    Returns:
        Configured root logger.
    """
    global _logger

    # Set default format (structured, informative)
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | "
            "%(message)s"
        )

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get or create root logger
    _logger = logging.getLogger("exoplanet_discovery")
    _logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    _logger.handlers.clear()

    # Console handler (always added)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        _logger.info(f"Logging to file: {log_file}")

    return _logger


def get_app_logger() -> logging.Logger:
    """Get the application's root logger (post-initialization).

    Returns:
        Configured application logger, or creates a default one if not initialized.
    """
    global _logger
    if _logger is None:
        _logger = configure_logging()
    return _logger


# Module-level convenience functions

def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a DEBUG-level message."""
    get_app_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an INFO-level message."""
    get_app_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a WARNING-level message."""
    get_app_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an ERROR-level message."""
    get_app_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a CRITICAL-level message."""
    get_app_logger().critical(msg, *args, **kwargs)
