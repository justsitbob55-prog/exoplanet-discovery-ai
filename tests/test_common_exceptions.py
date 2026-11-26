"""Unit tests for custom exception hierarchy."""

import pytest

from src.common import exceptions


class TestExceptionHierarchy:
    """Test custom exception class hierarchy and properties."""

    def test_base_exception_is_exception(self):
        """Test that ExoplanetDiscoveryError inherits from Exception."""
        assert issubclass(exceptions.ExoplanetDiscoveryError, Exception)

    def test_ingestion_error_hierarchy(self):
        """Test ingestion error hierarchy."""
        assert issubclass(exceptions.IngestionError, exceptions.ExoplanetDiscoveryError)
        assert issubclass(exceptions.InvalidDataSourceError, exceptions.IngestionError)
        assert issubclass(exceptions.DataDownloadError, exceptions.IngestionError)
        assert issubclass(exceptions.DataParseError, exceptions.IngestionError)

    def test_light_curve_error_hierarchy(self):
        """Test light curve error hierarchy."""
        assert issubclass(
            exceptions.InvalidLightCurveError,
            exceptions.ExoplanetDiscoveryError,
        )
        assert issubclass(
            exceptions.LightCurveProcessingError,
            exceptions.ExoplanetDiscoveryError,
        )

    def test_preprocessing_error_hierarchy(self):
        """Test preprocessing error hierarchy."""
        assert issubclass(exceptions.PreprocessingError, exceptions.ExoplanetDiscoveryError)
        assert issubclass(exceptions.NormalizationError, exceptions.PreprocessingError)
        assert issubclass(exceptions.DetrendingError, exceptions.PreprocessingError)
        assert issubclass(exceptions.InterpolationError, exceptions.PreprocessingError)

    def test_model_error_hierarchy(self):
        """Test model error hierarchy."""
        assert issubclass(exceptions.ModelError, exceptions.ExoplanetDiscoveryError)
        assert issubclass(exceptions.ModelLoadError, exceptions.ModelError)
        assert issubclass(exceptions.InferenceError, exceptions.ModelError)
        assert issubclass(exceptions.DetectionError, exceptions.ModelError)

    def test_configuration_error_hierarchy(self):
        """Test configuration error hierarchy."""
        assert issubclass(exceptions.ConfigurationError, exceptions.ExoplanetDiscoveryError)
        assert issubclass(
            exceptions.MissingConfigurationError,
            exceptions.ConfigurationError,
        )
        assert issubclass(
            exceptions.InvalidConfigurationError,
            exceptions.ConfigurationError,
        )

    def test_exception_can_be_raised_and_caught(self):
        """Test that exceptions can be raised and caught correctly."""
        with pytest.raises(exceptions.InvalidLightCurveError):
            raise exceptions.InvalidLightCurveError("Bad light curve")

        with pytest.raises(exceptions.ExoplanetDiscoveryError):
            raise exceptions.InvalidLightCurveError("Bad light curve")

    def test_exception_message_preservation(self):
        """Test that exception messages are preserved."""
        msg = "Test error message"
        try:
            raise exceptions.ConfigurationError(msg)
        except exceptions.ConfigurationError as e:
            assert str(e) == msg

    def test_exception_instantiation(self):
        """Test instantiation of all custom exceptions."""
        # Test each exception can be instantiated
        exceptions.InvalidDataSourceError("test")
        exceptions.DataDownloadError("test")
        exceptions.DataParseError("test")
        exceptions.InvalidLightCurveError("test")
        exceptions.LightCurveProcessingError("test")
        exceptions.PreprocessingError("test")
        exceptions.NormalizationError("test")
        exceptions.DetrendingError("test")
        exceptions.InterpolationError("test")
        exceptions.ModelError("test")
        exceptions.ModelLoadError("test")
        exceptions.InferenceError("test")
        exceptions.DetectionError("test")
        exceptions.ConfigurationError("test")
        exceptions.MissingConfigurationError("test")
        exceptions.InvalidConfigurationError("test")
