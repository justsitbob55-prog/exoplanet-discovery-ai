"""
Project-wide constants to avoid magic numbers and enable easy configuration.

All constants are organized by domain and use UPPER_CASE naming.
"""

# ============================================================================
# Astronomical Constants (SI units)
# ============================================================================

# Earth and Solar system reference values
EARTH_RADIUS_KM: float = 6371.0  # Earth radius in kilometers
EARTH_RADIUS_M: float = EARTH_RADIUS_KM * 1000.0  # Earth radius in meters
SOLAR_RADIUS_KM: float = 696000.0  # Solar radius in kilometers
SOLAR_RADIUS_M: float = SOLAR_RADIUS_KM * 1000.0  # Solar radius in meters

# Conversion factors
EARTH_RADII_TO_KM: float = 6371.0  # Convert Earth radii to km
SOLAR_RADII_TO_KM: float = 696000.0  # Convert Solar radii to km
JUPITER_RADIUS_KM: float = 69911.0  # Jupiter radius in km
JUPITER_RADII_TO_KM: float = JUPITER_RADIUS_KM

# Time conversions
SECONDS_PER_DAY: float = 86400.0
HOURS_PER_DAY: float = 24.0
MINUTES_PER_HOUR: float = 60.0

# ============================================================================
# Light Curve Constraints
# ============================================================================

# Valid flux ranges (normalized around 1.0)
MIN_VALID_FLUX: float = 0.0  # Minimum physical flux value
MAX_VALID_FLUX: float = 2.0  # Maximum reasonable normalized flux

# Transit/anomaly detection constraints
MIN_TRANSIT_DEPTH: float = 0.0001  # Minimum detectable transit depth (0.01%)
MAX_TRANSIT_DEPTH: float = 1.0  # Maximum transit depth (100% brightness loss)
MIN_TRANSIT_DURATION_HOURS: float = 0.5  # Minimum transit duration
MAX_TRANSIT_DURATION_HOURS: float = 100.0  # Maximum transit duration

# Signal-to-noise ratio thresholds
MIN_SNR_FOR_DETECTION: float = 3.0  # 3-sigma detection threshold
STRONG_SNR_THRESHOLD: float = 7.0  # Considered "strong" detection

# Confidence score range
MIN_CONFIDENCE: float = 0.0
MAX_CONFIDENCE: float = 1.0

# ============================================================================
# Data Processing Parameters
# ============================================================================

# Normalization
FLUX_NORMALIZATION_PERCENTILE: float = 95.0  # Use 95th percentile for normalization
SIGMA_CLIPPING_THRESHOLD: float = 5.0  # Number of sigma for outlier removal

# Detrending
POLYNOMIAL_DETREND_ORDER: int = 3  # Default polynomial degree for detrending
SAVGOL_WINDOW_LENGTH: int = 51  # Savitzky-Golay filter window length

# Feature extraction
N_FEATURES_BASE: int = 32  # Base number of features for embeddings
N_WAVELETS: int = 8  # Number of wavelet scales for analysis

# ============================================================================
# Model Parameters
# ============================================================================

# Training defaults
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 100
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_TRAIN_TEST_SPLIT: float = 0.8

# Model architecture
MODEL_EMBEDDING_DIM: int = 128
MODEL_HIDDEN_DIM: int = 256

# ============================================================================
# File and Data Formats
# ============================================================================

# Supported data formats
SUPPORTED_LIGHT_CURVE_FORMATS: list[str] = ["fits", "csv", "hdf5", "parquet"]
SUPPORTED_CONFIG_FORMATS: list[str] = ["yaml", "json"]

# File extensions
FITS_EXTENSION: str = ".fits"
CSV_EXTENSION: str = ".csv"
HDF5_EXTENSION: str = ".h5"
PARQUET_EXTENSION: str = ".parquet"
YAML_EXTENSION: str = ".yaml"
JSON_EXTENSION: str = ".json"

# ============================================================================
# Mission and Catalog Identifiers
# ============================================================================

# Space missions
MISSION_TESS: str = "TESS"
MISSION_KEPLER: str = "Kepler"
MISSION_K2: str = "K2"

# Common catalog identifiers
SUPPORTED_MISSIONS: list[str] = [MISSION_TESS, MISSION_KEPLER, MISSION_K2]

# ============================================================================
# Error and Warning Thresholds
# ============================================================================

# Missing/NaN data tolerance
MAX_NAN_FRACTION: float = 0.2  # Reject light curves with > 20% NaN
MAX_CONSECUTIVE_NANS: int = 100  # Max consecutive NaN points before rejection

# Photometric uncertainty
MAX_FLUX_ERROR_FRACTION: float = 0.1  # Max flux error as fraction of flux

# ============================================================================
# Output and Reporting
# ============================================================================

# Candidate disposition keywords
DISPOSITION_CANDIDATE: str = "CANDIDATE"
DISPOSITION_CONFIRMED: str = "CONFIRMED"
DISPOSITION_FALSE_POSITIVE: str = "FALSE_POSITIVE"
DISPOSITION_PLANETARY: str = "PLANETARY"
DISPOSITION_STELLAR: str = "STELLAR"
DISPOSITION_INSTRUMENTAL: str = "INSTRUMENTAL"

VALID_DISPOSITIONS: list[str] = [
    DISPOSITION_CANDIDATE,
    DISPOSITION_CONFIRMED,
    DISPOSITION_FALSE_POSITIVE,
    DISPOSITION_PLANETARY,
    DISPOSITION_STELLAR,
    DISPOSITION_INSTRUMENTAL,
]

# Confidence thresholds for classification
CONFIDENCE_THRESHOLD_HIGH: float = 0.9
CONFIDENCE_THRESHOLD_MEDIUM: float = 0.7
CONFIDENCE_THRESHOLD_LOW: float = 0.5
