"""Project-wide constants and configuration values.

This module centralizes magic numbers and configuration constants
to avoid hardcoding values throughout the codebase.
"""

# ============================================================================
# Physical/Astronomical Constants
# ============================================================================

# Earth/stellar reference values
EARTH_RADIUS_KM: float = 6371.0
"""Mean Earth radius in kilometers."""

EARTH_MASS_KG: float = 5.972e24
"""Earth mass in kilograms."""

SOLAR_RADIUS_KM: float = 696000.0
"""Solar radius in kilometers."""

SOLAR_MASS_KG: float = 1.989e30
"""Solar mass in kilograms."""

# Transit detection
MINIMUM_TRANSIT_DEPTH_PPM: float = 10.0
"""Minimum detectable transit depth in parts per million (ppm)."""

MINIMUM_TRANSIT_DURATION_MINUTES: float = 30.0
"""Minimum expected transit duration in minutes (for Earth-Sun analogs)."""

MAXIMUM_TRANSIT_DURATION_HOURS: float = 20.0
"""Maximum expected transit duration in hours (for Earth-Sun analogs)."""

# Light curve characteristics
MINIMUM_LIGHT_CURVE_CADENCES: int = 10
"""Minimum number of flux measurements for a valid light curve."""

MAXIMUM_MISSING_CADENCE_FRACTION: float = 0.5
"""Maximum fraction of missing/bad data (default allowance)."""

# ============================================================================
# Preprocessing Parameters
# ============================================================================

# Detrending
DEFAULT_DETREND_WINDOW_SIZE: int = 100
"""Default window size for rolling detrending (cadences)."""

DEFAULT_DETREND_POLY_ORDER: int = 2
"""Default polynomial order for polynomial detrending."""

# Normalization
DEFAULT_FLUX_NORMALIZATION_METHOD: str = "median"
"""Default flux normalization: 'median' or 'mean'."""

# ============================================================================
# Feature Extraction Parameters
# ============================================================================

# Period detection
MINIMUM_PERIOD_DAYS: float = 0.5
"""Minimum detectable orbital period (days)."""

MAXIMUM_PERIOD_DAYS: float = 365.0
"""Maximum assumed orbital period (days) for transit surveys."""

# Periodogram/FFT
DEFAULT_FREQUENCY_RESOLUTION_UHERTZ: float = 1.0
"""Default frequency resolution in micro-Hertz for periodograms."""

# ============================================================================
# Model/Detection Parameters
# ============================================================================

# Probability thresholds
DEFAULT_EXOPLANET_CONFIDENCE_THRESHOLD: float = 0.5
"""Default probability threshold for exoplanet classification."""

HIGH_CONFIDENCE_THRESHOLD: float = 0.9
"""High-confidence threshold for publication-ready candidates."""

LOW_CONFIDENCE_THRESHOLD: float = 0.3
"""Low-confidence threshold for follow-up observations."""

# Signal-to-noise ratio
MINIMUM_SNR_FOR_DETECTION: float = 5.0
"""Minimum SNR required for a signal to be considered a detection."""

# ============================================================================
# Data Processing
# ============================================================================

# Batch processing
DEFAULT_BATCH_SIZE: int = 32
"""Default batch size for processing light curves."""

# Random seed for reproducibility
RANDOM_SEED: int = 42
"""Global random seed for reproducibility."""

# ============================================================================
# API/Application Parameters
# ============================================================================

# Application defaults
DEFAULT_LOG_LEVEL: str = "INFO"
"""Default logging level."""

DEFAULT_MAX_UPLOAD_SIZE_MB: float = 100.0
"""Maximum file upload size in MB for API."""

# ============================================================================
# File Paths and Extensions
# ============================================================================

DEFAULT_CONFIG_FILE: str = "configs/default_config.yaml"
"""Default configuration file path (relative to project root)."""

LIGHT_CURVE_FILE_EXTENSIONS: tuple[str, ...] = (".fits", ".fits.gz", ".hdf5", ".csv")
"""Supported light curve file formats."""

MODEL_CHECKPOINT_EXTENSION: str = ".pth"
"""Standard model checkpoint file extension."""
