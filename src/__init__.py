"""
Exoplanet Discovery & Light-Curve Anomaly Detector

AI/ML tools for discovering exoplanets and detecting anomalies in light curves.

Version: 0.1.0
"""

from src.common import (
    ConfigurationError,
    DataDownloadError,
    DataParseError,
    DetectionError,
    DetrendingError,
    ExoplanetCandidate,
    ExoplanetDiscoveryError,
    InferenceError,
    IngestionError,
    InterpolationError,
    InvalidConfigurationError,
    InvalidDataSourceError,
    InvalidLightCurveError,
    LightCurve,
    LightCurveProcessingError,
    MissingConfigurationError,
    ModelError,
    ModelLoadError,
    NormalizationError,
    PreprocessingError,
    TransitEvent,
    configure_root_logging,
    get_logger,
    set_logger_level,
)
from src.config import ProjectConfig

__version__ = "0.1.0"

__all__ = [
    # Models
    "LightCurve",
    "TransitEvent",
    "ExoplanetCandidate",
    # Configuration
    "ProjectConfig",
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
