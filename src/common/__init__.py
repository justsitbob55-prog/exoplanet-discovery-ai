"""
Common utilities, models, and infrastructure for the exoplanet discovery project.

This package provides:
- Data models (LightCurve, TransitEvent, ExoplanetCandidate)
- Custom exception types
- Centralized logging
- Project-wide constants
"""

from src.common.exceptions import (
    ConfigurationError,
    DataDownloadError,
    DataParseError,
    DetectionError,
    DetrendingError,
    ExoplanetDiscoveryError,
    InferenceError,
    IngestionError,
    InterpolationError,
    InvalidConfigurationError,
    InvalidDataSourceError,
    InvalidLightCurveError,
    LightCurveProcessingError,
    MissingConfigurationError,
    ModelError,
    ModelLoadError,
    NormalizationError,
    PreprocessingError,
)
from src.common.logging import configure_root_logging, get_logger, set_logger_level
from src.common.models import ExoplanetCandidate, LightCurve, TransitEvent

__all__ = [
    # Models
    "LightCurve",
    "TransitEvent",
    "ExoplanetCandidate",
    # Exceptions
    "ExoplanetDiscoveryError",
    "IngestionError",
    "InvalidDataSourceError",
    "DataDownloadError",
    "DataParseError",
    "InvalidLightCurveError",
    "LightCurveProcessingError",
    "PreprocessingError",
    "NormalizationError",
    "DetrendingError",
    "InterpolationError",
    "ModelError",
    "ModelLoadError",
    "InferenceError",
    "DetectionError",
    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    # Logging
    "get_logger",
    "configure_root_logging",
    "set_logger_level",
]
