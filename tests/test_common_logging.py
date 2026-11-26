"""Unit tests for logging configuration."""

import logging

from src.common.logging import configure_root_logging, get_logger, set_logger_level


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_same_name_returns_same_logger(self):
        """Test that requesting the same logger name returns the same instance."""
        logger1 = get_logger("test_same_name")
        logger2 = get_logger("test_same_name")
        assert logger1 is logger2

    def test_logger_has_handler(self):
        """Test that returned logger has at least one handler configured."""
        logger = get_logger("test_with_handler")
        # Clear any existing handlers first
        logger.handlers.clear()
        logger = get_logger("test_with_handler_fresh")
        assert len(logger.handlers) > 0

    def test_logger_level_is_debug(self):
        """Test that logger is set to DEBUG level."""
        logger = get_logger("test_level_debug")
        logger.handlers.clear()  # Reset
        logger = get_logger("test_level_debug_fresh")
        assert logger.level == logging.DEBUG

    def test_different_logger_names(self):
        """Test that different logger names create different loggers."""
        logger1 = get_logger("logger_one")
        logger2 = get_logger("logger_two")
        assert logger1 is not logger2
        assert logger1.name == "logger_one"
        assert logger2.name == "logger_two"


class TestConfigureRootLogging:
    """Test suite for configure_root_logging function."""

    def test_configure_root_logging_sets_level(self):
        """Test that configure_root_logging sets root logger level."""
        root = logging.getLogger()
        root.handlers.clear()

        configure_root_logging(level=logging.WARNING)

        assert root.level == logging.WARNING

    def test_configure_root_logging_adds_handler(self):
        """Test that configure_root_logging adds a handler to root logger."""
        root = logging.getLogger()

        # Clear and reconfigure
        root.handlers.clear()
        configure_root_logging()

        assert len(root.handlers) > 0

    def test_configure_root_logging_default_level(self):
        """Test that default level is INFO."""
        root = logging.getLogger()
        root.handlers.clear()

        configure_root_logging()

        assert root.level == logging.INFO


class TestSetLoggerLevel:
    """Test suite for set_logger_level function."""

    def test_set_logger_level_by_name(self):
        """Test setting level for a named logger."""
        logger = get_logger("test_set_level")
        set_logger_level("test_set_level", logging.ERROR)
        assert logger.level == logging.ERROR

    def test_set_root_logger_level(self):
        """Test setting level for root logger when name is None."""
        root = logging.getLogger()
        set_logger_level(None, logging.CRITICAL)
        assert root.level == logging.CRITICAL

    def test_set_logger_level_affects_handlers(self):
        """Test that set_logger_level updates handler levels."""
        logger = get_logger("test_handler_level")
        set_logger_level("test_handler_level", logging.DEBUG)

        # Check that handlers have been updated
        for handler in logger.handlers:
            assert handler.level == logging.DEBUG


class TestLoggerFormatting:
    """Test suite for logger output formatting."""

    def test_logger_produces_output(self, caplog):
        """Test that logger produces formatted output."""
        logger = get_logger("test_format_output")

        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        assert "Test message" in caplog.text
        assert "test_format_output" in caplog.text
