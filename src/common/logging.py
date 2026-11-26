"""
Centralized logging configuration for the exoplanet discovery project.

Provides structured, application-wide logging with console output.
Uses Python's standard logging module without external dependencies.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with standardized configuration.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler with structured formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Format: timestamp | level | logger_name | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def configure_root_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger for application startup.

    Useful for one-time application-wide configuration.

    Args:
        level: Logging level (default INFO).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def set_logger_level(name: str | None = None, level: int = logging.INFO) -> None:
    """
    Set logging level for a specific logger or root logger.

    Args:
        name: Logger name. If None, configures root logger.
        level: Logging level to set.
    """
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)

    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
