"""Unit tests for exoplanet_discovery.common.exceptions.

Tests verify that all custom exceptions:
- Inherit from correct base class
- Can be raised and caught properly
- Have proper exception hierarchy
"""

import pytest

from exoplanet_discovery.common.exceptions import (
    ConfigurationError,
    ExoplanetDiscoveryError,
    FeatureExtractionError,
    IngestionError,
    InvalidLightCurveError,
    ModelError,
    PreprocessingError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy and inheritance."""

    def test_all_exceptions_inherit_from_base(self) -> None:
        """Verify all custom exceptions inherit from ExoplanetDiscoveryError."""
        custom_exceptions = [
            InvalidLightCurveError,
            IngestionError,
            PreprocessingError,
            FeatureExtractionError,
            ModelError,
            ConfigurationError,
        ]

        for exc_class in custom_exceptions:
            assert issubclass(exc_class, ExoplanetDiscoveryError)

    def test_base_exception_inherits_from_exception(self) -> None:
        """Verify base exception inherits from Exception."""
        assert issubclass(ExoplanetDiscoveryError, Exception)

    def test_invalid_light_curve_error_can_be_raised(self) -> None:
        """Test raising and catching InvalidLightCurveError."""
        with pytest.raises(InvalidLightCurveError):
            raise InvalidLightCurveError("Bad light curve data")

    def test_ingestion_error_can_be_raised(self) -> None:
        """Test raising and catching IngestionError."""
        with pytest.raises(IngestionError):
            raise IngestionError("Failed to fetch data")

    def test_preprocessing_error_can_be_raised(self) -> None:
        """Test raising and catching PreprocessingError."""
        with pytest.raises(PreprocessingError):
            raise PreprocessingError("Detrending failed")

    def test_feature_extraction_error_can_be_raised(self) -> None:
        """Test raising and catching FeatureExtractionError."""
        with pytest.raises(FeatureExtractionError):
            raise FeatureExtractionError("Period detection failed")

    def test_model_error_can_be_raised(self) -> None:
        """Test raising and catching ModelError."""
        with pytest.raises(ModelError):
            raise ModelError("Model checkpoint not found")

    def test_configuration_error_can_be_raised(self) -> None:
        """Test raising and catching ConfigurationError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid config parameter")

    def test_catch_with_base_class(self) -> None:
        """Test catching specific exceptions with base class."""
        with pytest.raises(ExoplanetDiscoveryError):
            raise InvalidLightCurveError("Test error")

        with pytest.raises(ExoplanetDiscoveryError):
            raise ModelError("Test error")

    def test_exception_message_preserved(self) -> None:
        """Verify exception message is preserved."""
        msg = "Test error message"
        try:
            raise InvalidLightCurveError(msg)
        except InvalidLightCurveError as e:
            assert str(e) == msg

    def test_exception_chaining(self) -> None:
        """Test exception chaining with from."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ModelError("Model failed") from e
        except ModelError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
