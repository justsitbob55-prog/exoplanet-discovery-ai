"""Custom exceptions for the exoplanet discovery package.

This module defines project-specific exception types to enable precise
error handling and meaningful error messages across the pipeline.
"""


class ExoplanetDiscoveryError(Exception):
    """Base exception for all exoplanet discovery errors."""

    pass


class InvalidLightCurveError(ExoplanetDiscoveryError):
    """Raised when light curve data is invalid or malformed.

    Examples:
        - Mismatched array lengths
        - Missing required fields
        - Non-finite values (NaN, inf)
        - Insufficient data points
    """

    pass


class IngestionError(ExoplanetDiscoveryError):
    """Raised when data ingestion fails.

    Examples:
        - Cannot fetch from data source
        - File format not recognized
        - Network timeout
        - Authentication failure
    """

    pass


class PreprocessingError(ExoplanetDiscoveryError):
    """Raised when light curve preprocessing fails.

    Examples:
        - Detrending algorithm did not converge
        - Gap filling failed
        - Invalid preprocessing parameters
    """

    pass


class FeatureExtractionError(ExoplanetDiscoveryError):
    """Raised when feature extraction fails.

    Examples:
        - Period detection did not converge
        - Invalid signal characteristics
        - Insufficient data for feature computation
    """

    pass


class ModelError(ExoplanetDiscoveryError):
    """Raised when model loading, initialization, or inference fails.

    Examples:
        - Model checkpoint not found
        - Checkpoint format incompatible
        - Input shape mismatch
        - Model inference failed
    """

    pass


class ConfigurationError(ExoplanetDiscoveryError):
    """Raised when configuration is invalid or inconsistent.

    Examples:
        - Missing required config keys
        - Invalid parameter values
        - Conflicting settings
        - Config file not found
    """

    pass
