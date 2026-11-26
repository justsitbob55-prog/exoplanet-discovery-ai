"""Configuration system using Pydantic for type-safe config management.

Provides structured configuration models for different pipeline stages:
- DataConfig: Data ingestion and storage
- PreprocessingConfig: Light curve preprocessing parameters
- FeatureConfig: Feature extraction configuration
- DetectionConfig: Anomaly/transit detection configuration
- TrainingConfig: Model training parameters
- AppConfig: Application-wide configuration

Supports loading from YAML files via external tools.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class DataConfig(BaseModel):
    """Configuration for data ingestion and handling.

    Attributes:
        source: Data source ('Kepler', 'TESS', 'synthetic', etc.).
        raw_data_dir: Path to raw data directory.
        processed_data_dir: Path to processed data directory.
        max_objects: Maximum number of objects to process (None = all).
        validate_on_load: Whether to validate data on ingestion.
    """

    source: str = "TESS"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    max_objects: int | None = None
    validate_on_load: bool = True

    class Config:
        """Pydantic config."""

        extra = "forbid"


class PreprocessingConfig(BaseModel):
    """Configuration for light curve preprocessing.

    Attributes:
        detrend_enabled: Whether to apply detrending.
        detrend_method: Method ('polynomial', 'lowess', 'savitzky_golay').
        detrend_window_size: Window size for detrending (cadences).
        detrend_poly_order: Polynomial order (for polynomial detrending).
        normalize_enabled: Whether to normalize flux.
        normalization_method: Method ('median', 'mean').
        gap_fill_enabled: Whether to interpolate gaps.
        gap_fill_method: Method ('linear', 'cubic').
    """

    detrend_enabled: bool = True
    detrend_method: str = "polynomial"
    detrend_window_size: int = 100
    detrend_poly_order: int = 2
    normalize_enabled: bool = True
    normalization_method: str = "median"
    gap_fill_enabled: bool = False
    gap_fill_method: str = "linear"

    @field_validator("detrend_method")
    @classmethod
    def validate_detrend_method(cls, v: str) -> str:
        """Validate detrending method."""
        valid = {"polynomial", "lowess", "savitzky_golay"}
        if v not in valid:
            raise ValueError(f"detrend_method must be one of {valid}, got {v}")
        return v

    @field_validator("normalization_method")
    @classmethod
    def validate_normalization_method(cls, v: str) -> str:
        """Validate normalization method."""
        valid = {"median", "mean"}
        if v not in valid:
            raise ValueError(f"normalization_method must be one of {valid}, got {v}")
        return v

    class Config:
        """Pydantic config."""

        extra = "forbid"


class FeatureConfig(BaseModel):
    """Configuration for feature extraction.

    Attributes:
        extract_periodicity_features: Extract period/periodogram features.
        extract_transit_features: Extract transit-specific features.
        extract_statistical_features: Extract statistical features.
        min_period_days: Minimum period to search (days).
        max_period_days: Maximum period to search (days).
        frequency_resolution_uhertz: Frequency resolution (micro-Hertz).
    """

    extract_periodicity_features: bool = True
    extract_transit_features: bool = True
    extract_statistical_features: bool = True
    min_period_days: float = 0.5
    max_period_days: float = 365.0
    frequency_resolution_uhertz: float = 1.0

    @field_validator("min_period_days", "max_period_days")
    @classmethod
    def validate_periods_positive(cls, v: float) -> float:
        """Ensure periods are positive."""
        if v <= 0:
            raise ValueError(f"Period must be positive, got {v}")
        return v

    @field_validator("max_period_days")
    @classmethod
    def validate_max_min_period(cls, v: float, info: ValidationInfo) -> float:
        """Ensure max >= min period."""
        data = info.data
        if "min_period_days" in data and v < data["min_period_days"]:
            raise ValueError(
                f"max_period_days ({v}) must be >= min_period_days "
                f"({data['min_period_days']})"
            )
        return v

    class Config:
        """Pydantic config."""

        extra = "forbid"


class DetectionConfig(BaseModel):
    """Configuration for transit/anomaly detection.

    Attributes:
        detector_type: Type of detector ('classical', 'deep_learning').
        probability_threshold: Confidence threshold for classification [0, 1].
        min_snr: Minimum SNR for a detection to be reported.
        model_checkpoint: Path to model checkpoint (if using DL).
    """

    detector_type: str = "classical"
    probability_threshold: float = 0.5
    min_snr: float = 5.0
    model_checkpoint: str | None = None

    @field_validator("probability_threshold")
    @classmethod
    def validate_probability_threshold(cls, v: float) -> float:
        """Ensure threshold in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError(f"probability_threshold must be in [0, 1], got {v}")
        return v

    @field_validator("detector_type")
    @classmethod
    def validate_detector_type(cls, v: str) -> str:
        """Validate detector type."""
        valid = {"classical", "deep_learning"}
        if v not in valid:
            raise ValueError(f"detector_type must be one of {valid}, got {v}")
        return v

    class Config:
        """Pydantic config."""

        extra = "forbid"


class TrainingConfig(BaseModel):
    """Configuration for model training.

    Attributes:
        model_type: Model architecture ('random_forest', 'xgboost', 'cnn', etc.).
        epochs: Number of training epochs (for deep learning).
        batch_size: Batch size for training.
        learning_rate: Learning rate (for deep learning).
        validation_split: Fraction of data for validation [0, 1].
        random_seed: Random seed for reproducibility.
    """

    model_type: str = "random_forest"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    random_seed: int = 42

    @field_validator("epochs", "batch_size")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v <= 0:
            raise ValueError(f"Must be positive, got {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Ensure valid learning rate."""
        if v <= 0 or v >= 1:
            raise ValueError(f"learning_rate must be in (0, 1), got {v}")
        return v

    @field_validator("validation_split")
    @classmethod
    def validate_validation_split(cls, v: float) -> float:
        """Ensure validation_split in (0, 1)."""
        if not 0 < v < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {v}")
        return v

    class Config:
        """Pydantic config."""

        extra = "forbid"


class AppConfig(BaseModel):
    """Top-level application configuration.

    Attributes:
        data: Data configuration.
        preprocessing: Preprocessing configuration.
        features: Feature extraction configuration.
        detection: Detection configuration.
        training: Training configuration.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path.
    """

    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    log_level: str = "INFO"
    log_file: str | None = None

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v}")
        return v

    class Config:
        """Pydantic config."""

        extra = "forbid"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "AppConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Loaded AppConfig instance.

        Raises:
            FileNotFoundError: If YAML file not found.
            ValueError: If YAML is invalid or config validation fails.
        """
        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}") from e

        return cls(**data)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Write configuration to YAML file.

        Args:
            yaml_path: Path to write YAML configuration.
        """
        import yaml

        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return self.model_dump()
