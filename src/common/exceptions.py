"""
Custom exceptions for the exoplanet discovery project.

Organized by domain (ingestion, preprocessing, model/inference, configuration).
"""


class ExoplanetDiscoveryError(Exception):
    """Base exception for all project-specific errors."""

    pass


# Ingestion-related exceptions


class IngestionError(ExoplanetDiscoveryError):
    """Base exception for data ingestion failures."""

    pass


class InvalidDataSourceError(IngestionError):
    """Raised when a data source is unavailable, invalid, or inaccessible."""

    pass


class DataDownloadError(IngestionError):
    """Raised when downloading remote data fails."""

    pass


class DataParseError(IngestionError):
    """Raised when parsing data format fails (malformed FITS, CSV, etc.)."""

    pass


# Light curve data validation


class InvalidLightCurveError(ExoplanetDiscoveryError):
    """Raised when a light curve fails validation (NaN, mismatched shapes, etc.)."""

    pass


class LightCurveProcessingError(ExoplanetDiscoveryError):
    """Raised during light curve processing (normalization, detrending, etc.)."""

    pass


# Preprocessing exceptions


class PreprocessingError(ExoplanetDiscoveryError):
    """Base exception for preprocessing pipeline failures."""

    pass


class NormalizationError(PreprocessingError):
    """Raised when flux normalization fails."""

    pass


class DetrendingError(PreprocessingError):
    """Raised when detrending/systematics removal fails."""

    pass


class InterpolationError(PreprocessingError):
    """Raised when gap-filling or interpolation fails."""

    pass


# Model/Inference exceptions


class ModelError(ExoplanetDiscoveryError):
    """Base exception for model/inference failures."""

    pass


class ModelLoadError(ModelError):
    """Raised when loading a trained model fails."""

    pass


class InferenceError(ModelError):
    """Raised when model inference fails."""

    pass


class DetectionError(ModelError):
    """Raised when transit/anomaly detection fails."""

    pass


# Configuration exceptions


class ConfigurationError(ExoplanetDiscoveryError):
    """Base exception for configuration-related errors."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration validation fails."""

    pass
