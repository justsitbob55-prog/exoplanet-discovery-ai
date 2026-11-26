"""Unit tests for exoplanet_discovery.common.logging.

Tests verify that:
- Logging can be configured properly
- Loggers can be created and used
- Console output works
- File logging works (when requested)
- Log levels are respected
"""

import logging
import tempfile
from pathlib import Path

from exoplanet_discovery.common import logging as app_logging


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """get_logger should return a Logger instance."""
        logger = app_logging.get_logger(__name__)
        assert isinstance(logger, logging.Logger)

    def test_get_logger_same_name_returns_same_instance(self) -> None:
        """Multiple calls with same name should return same logger."""
        name = "test.logger"
        logger1 = app_logging.get_logger(name)
        logger2 = app_logging.get_logger(name)
        assert logger1 is logger2

    def test_get_logger_different_names_different_instances(self) -> None:
        """Different names should return different loggers."""
        logger1 = app_logging.get_logger("logger1")
        logger2 = app_logging.get_logger("logger2")
        assert logger1 is not logger2


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_returns_logger(self) -> None:
        """configure_logging should return a Logger instance."""
        logger = app_logging.configure_logging(level=logging.INFO)
        assert isinstance(logger, logging.Logger)

    def test_configure_logging_default_level(self) -> None:
        """Default logging level should be INFO."""
        logger = app_logging.configure_logging()
        assert logger.level == logging.INFO

    def test_configure_logging_custom_level(self) -> None:
        """Should set custom logging level."""
        logger = app_logging.configure_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_configure_logging_adds_console_handler(self) -> None:
        """Should add a console handler."""
        logger = app_logging.configure_logging()
        # Check that at least one handler exists
        assert len(logger.handlers) > 0
        # Check that at least one handler is StreamHandler
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        )
        assert has_stream_handler

    def test_configure_logging_with_file(self) -> None:
        """Should add file handler when log_file is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = app_logging.configure_logging(log_file=log_file)

            # Check that file handler was added
            has_file_handler = any(
                isinstance(h, logging.FileHandler) for h in logger.handlers
            )
            assert has_file_handler

            # Log a message and verify it was written
            logger.info("Test log message")
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test log message" in content

            # Clean up handlers to release file handle
            for handler in logger.handlers:
                handler.close()

    def test_configure_logging_custom_format(self) -> None:
        """Should accept custom format string."""
        custom_format = "%(levelname)s: %(message)s"
        logger = app_logging.configure_logging(format_string=custom_format)

        # Verify format was applied to handlers
        for handler in logger.handlers:
            if handler.formatter:
                # The format string should be in the formatter's format
                assert handler.formatter._fmt == custom_format

    def test_configure_logging_removes_old_handlers(self) -> None:
        """Multiple calls should not accumulate handlers."""
        app_logging.configure_logging(level=logging.INFO)
        initial_count = len(app_logging.get_app_logger().handlers)

        app_logging.configure_logging(level=logging.DEBUG)
        final_count = len(app_logging.get_app_logger().handlers)

        # Should have similar number of handlers (not accumulated)
        assert final_count <= initial_count + 1


class TestGetAppLogger:
    """Tests for get_app_logger function."""

    def test_get_app_logger_returns_logger(self) -> None:
        """get_app_logger should return a Logger instance."""
        logger = app_logging.get_app_logger()
        assert isinstance(logger, logging.Logger)

    def test_get_app_logger_initializes_if_needed(self) -> None:
        """Should initialize logger if not already done."""
        # Reset the global logger
        app_logging._logger = None

        logger = app_logging.get_app_logger()
        assert logger is not None
        assert isinstance(logger, logging.Logger)


class TestModuleLevelConvenienceFunctions:
    """Tests for module-level convenience logging functions."""

    def test_debug_function(self) -> None:
        """debug() should log at DEBUG level."""
        app_logging.configure_logging(level=logging.DEBUG)
        # Just verify it doesn't raise an exception
        app_logging.debug("Debug message")

    def test_info_function(self) -> None:
        """info() should log at INFO level."""
        app_logging.configure_logging(level=logging.INFO)
        app_logging.info("Info message")

    def test_warning_function(self) -> None:
        """warning() should log at WARNING level."""
        app_logging.configure_logging(level=logging.WARNING)
        app_logging.warning("Warning message")

    def test_error_function(self) -> None:
        """error() should log at ERROR level."""
        app_logging.configure_logging(level=logging.ERROR)
        app_logging.error("Error message")

    def test_critical_function(self) -> None:
        """critical() should log at CRITICAL level."""
        app_logging.configure_logging(level=logging.CRITICAL)
        app_logging.critical("Critical message")

    def test_convenience_functions_with_formatting(self) -> None:
        """Convenience functions should support message formatting."""
        app_logging.configure_logging()
        # Should not raise an exception
        app_logging.info("Message with %s", "formatting")
        app_logging.debug("Message with %d items", 42)


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_full_logging_workflow(self) -> None:
        """Test complete logging workflow with console logging."""
        # Configure logging (console only to avoid Windows file handle issues)
        logger = app_logging.configure_logging(level=logging.INFO)

        # Get a module logger
        module_logger = app_logging.get_logger("test_module")

        # Log at different levels (just verify no exceptions)
        module_logger.debug("Debug message (should not appear)")
        module_logger.info("Info message")
        module_logger.warning("Warning message")

        # Verify logger is properly configured
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_structured_logging_format(self) -> None:
        """Test that structured logging format is properly configured."""
        # Configure logging with custom format
        custom_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        logger = app_logging.configure_logging(
            level=logging.INFO, format_string=custom_format
        )

        # Verify format was applied
        assert logger.level == logging.INFO
        for handler in logger.handlers:
            if handler.formatter and handler.formatter._fmt is not None:
                assert custom_format in handler.formatter._fmt
