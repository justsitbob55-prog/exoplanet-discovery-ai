"""
Configuration system for the exoplanet discovery project.

Uses Pydantic v2 for validation and provides sensible defaults.
Configuration can be loaded from YAML or constructed programmatically.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data ingestion and storage."""

    raw_data_dir: Path = Field(
        default=Path("data/raw"),
        description="Path to raw downloaded light-curve data",
    )
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        description="Path to preprocessed, model-ready data",
    )
    external_data_dir: Path = Field(
        default=Path("data/external"),
        description="Path to external catalogs and metadata",
    )
    max_nan_fraction: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Reject light curves with NaN fraction exceeding this value",
    )
    max_consecutive_nans: int = Field(
        default=100,
        ge=1,
        description="Maximum consecutive NaN points before rejecting light curve",
    )

    @field_validator("raw_data_dir", "processed_data_dir", "external_data_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v


class PreprocessingConfig(BaseModel):
    """Configuration for light-curve preprocessing."""

    normalize: bool = Field(
        default=True,
        description="Apply flux normalization",
    )
    normalization_percentile: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Percentile to use for normalization",
    )
    detrend: bool = Field(
        default=True,
        description="Apply polynomial detrending to remove systematic trends",
    )
    polynomial_order: int = Field(
        default=3,
        ge=1,
        description="Polynomial degree for detrending",
    )
    sigma_clip: bool = Field(
        default=True,
        description="Apply sigma clipping to remove outliers",
    )
    sigma_threshold: float = Field(
        default=5.0,
        gt=0.0,
        description="Number of standard deviations for outlier detection",
    )
    interpolate_gaps: bool = Field(
        default=True,
        description="Interpolate NaN gaps in light curves",
    )
    outlier_sigma: float = Field(
        default=5.0,
        gt=0.0,
        description="Robust MAD threshold for rejecting flux outliers",
    )
    detrend_window: int = Field(
        default=101,
        ge=3,
        description="Savitzky-Golay window length, rounded to an odd value",
    )
    noise_reduction_enabled: bool = Field(
        default=False,
        description="Apply light smoothing after detrending",
    )


class FeatureExtractionConfig(BaseModel):
    """Configuration for feature extraction from light curves."""

    n_features: int = Field(
        default=32,
        ge=8,
        description="Number of features to extract per light curve",
    )
    n_wavelets: int = Field(
        default=8,
        ge=1,
        description="Number of wavelet scales for wavelet decomposition",
    )
    use_statistical_features: bool = Field(
        default=True,
        description="Extract statistical features (mean, std, skew, etc.)",
    )
    use_frequency_features: bool = Field(
        default=True,
        description="Extract frequency-domain features (periodogram, etc.)",
    )
    use_morphological_features: bool = Field(
        default=True,
        description="Extract morphological features from light curves",
    )


class DetectionConfig(BaseModel):
    """Configuration for transit/anomaly detection."""

    min_snr: float = Field(
        default=3.0,
        gt=0.0,
        description="Minimum signal-to-noise ratio for detection threshold",
    )
    min_transit_depth: float = Field(
        default=0.0001,
        ge=0.0,
        le=1.0,
        description="Minimum detectable transit depth (fraction)",
    )
    max_transit_depth: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum transit depth (fraction)",
    )
    min_transit_duration_hours: float = Field(
        default=0.5,
        gt=0.0,
        description="Minimum transit duration in hours",
    )
    max_transit_duration_hours: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum transit duration in hours",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for accepting detections",
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    batch_size: int = Field(
        default=32,
        ge=1,
        description="Training batch size",
    )
    epochs: int = Field(
        default=100,
        ge=1,
        description="Number of training epochs",
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Optimizer learning rate",
    )
    train_test_split: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Fraction of data used for training vs. testing",
    )
    validation_split: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fraction of training data for validation",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    checkpoint_dir: Path = Field(
        default=Path("models/checkpoints"),
        description="Directory to save model checkpoints",
    )
    early_stopping_patience: int = Field(
        default=10,
        ge=1,
        description="Number of epochs with no improvement before early stopping",
    )

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def convert_checkpoint_dir(cls, v: str | Path) -> Path:
        """Convert checkpoint_dir to Path object."""
        return Path(v) if isinstance(v, str) else v


class ProjectConfig(BaseModel):
    """
    Top-level configuration for the entire exoplanet discovery project.

    Aggregates all sub-configurations for easy management and validation.
    """

    project_name: str = Field(
        default="exoplanet-discovery",
        description="Project name",
    )
    version: str = Field(
        default="0.1.0",
        description="Project version",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging and extra diagnostics",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data ingestion and storage configuration",
    )
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing pipeline configuration",
    )
    feature_extraction: FeatureExtractionConfig = Field(
        default_factory=FeatureExtractionConfig,
        description="Feature extraction configuration",
    )
    detection: DetectionConfig = Field(
        default_factory=DetectionConfig,
        description="Transit/anomaly detection configuration",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Model training configuration",
    )

    class Config:
        """Pydantic config."""

        extra = "forbid"  # Disallow extra fields

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return self.model_dump()

    def to_yaml(self) -> str:
        """Serialize configuration to YAML string."""
        data = self.to_dict()
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ProjectConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ProjectConfig":
        """Load configuration from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            config_dict: Any = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        return cls(**config_dict)

    @classmethod
    def default(cls) -> "ProjectConfig":
        """Create configuration with all defaults."""
        return cls()
