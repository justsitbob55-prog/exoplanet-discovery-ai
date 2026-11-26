"""Unit tests for configuration system.

Tests verify that:
- Configuration models validate inputs correctly
- Invalid configurations are rejected
- Defaults are sensible
- YAML loading/saving works
- Sub-configurations are properly nested
"""

from pathlib import Path

import pytest

from src.config import (
    DataConfig,
    DetectionConfig,
    FeatureExtractionConfig,
    PreprocessingConfig,
    ProjectConfig,
    TrainingConfig,
)


class TestDataConfig:
    """Test DataConfig validation and functionality."""

    def test_default_data_config(self):
        """Test creating DataConfig with defaults."""
        config = DataConfig()

        assert config.raw_data_dir == Path("data/raw")
        assert config.processed_data_dir == Path("data/processed")
        assert config.external_data_dir == Path("data/external")
        assert config.max_nan_fraction == 0.2
        assert config.max_consecutive_nans == 100

    def test_data_config_string_paths_converted(self):
        """Test that string paths are converted to Path objects."""
        config = DataConfig(
            raw_data_dir="custom/raw",
            processed_data_dir="custom/processed",
        )

        assert isinstance(config.raw_data_dir, Path)
        assert isinstance(config.processed_data_dir, Path)
        assert config.raw_data_dir == Path("custom/raw")

    def test_data_config_nan_fraction_validation(self):
        """Test that max_nan_fraction is validated."""
        with pytest.raises(ValueError):
            DataConfig(max_nan_fraction=1.5)  # > 1.0

        with pytest.raises(ValueError):
            DataConfig(max_nan_fraction=-0.1)  # < 0.0

    def test_data_config_consecutive_nans_validation(self):
        """Test that max_consecutive_nans must be >= 1."""
        with pytest.raises(ValueError):
            DataConfig(max_consecutive_nans=0)

        config = DataConfig(max_consecutive_nans=1)
        assert config.max_consecutive_nans == 1


class TestPreprocessingConfig:
    """Test PreprocessingConfig validation and functionality."""

    def test_default_preprocessing_config(self):
        """Test creating PreprocessingConfig with defaults."""
        config = PreprocessingConfig()

        assert config.normalize is True
        assert config.normalization_percentile == 95.0
        assert config.detrend is True
        assert config.polynomial_order == 3
        assert config.sigma_clip is True
        assert config.sigma_threshold == 5.0
        assert config.interpolate_gaps is True

    def test_preprocessing_normalization_percentile_validation(self):
        """Test normalization_percentile is in [0, 100]."""
        with pytest.raises(ValueError):
            PreprocessingConfig(normalization_percentile=150.0)

        with pytest.raises(ValueError):
            PreprocessingConfig(normalization_percentile=-1.0)

    def test_preprocessing_polynomial_order_validation(self):
        """Test polynomial_order must be >= 1."""
        with pytest.raises(ValueError):
            PreprocessingConfig(polynomial_order=0)

    def test_preprocessing_sigma_threshold_validation(self):
        """Test sigma_threshold must be > 0."""
        with pytest.raises(ValueError):
            PreprocessingConfig(sigma_threshold=0.0)

        with pytest.raises(ValueError):
            PreprocessingConfig(sigma_threshold=-1.0)


class TestFeatureExtractionConfig:
    """Test FeatureExtractionConfig validation and functionality."""

    def test_default_feature_extraction_config(self):
        """Test creating FeatureExtractionConfig with defaults."""
        config = FeatureExtractionConfig()

        assert config.n_features == 32
        assert config.n_wavelets == 8
        assert config.use_statistical_features is True
        assert config.use_frequency_features is True
        assert config.use_morphological_features is True

    def test_n_features_minimum_validation(self):
        """Test n_features must be >= 8."""
        with pytest.raises(ValueError):
            FeatureExtractionConfig(n_features=7)

        config = FeatureExtractionConfig(n_features=8)
        assert config.n_features == 8

    def test_n_wavelets_minimum_validation(self):
        """Test n_wavelets must be >= 1."""
        with pytest.raises(ValueError):
            FeatureExtractionConfig(n_wavelets=0)


class TestDetectionConfig:
    """Test DetectionConfig validation and functionality."""

    def test_default_detection_config(self):
        """Test creating DetectionConfig with defaults."""
        config = DetectionConfig()

        assert config.min_snr == 3.0
        assert config.min_transit_depth == 0.0001
        assert config.max_transit_depth == 1.0
        assert config.min_transit_duration_hours == 0.5
        assert config.max_transit_duration_hours == 100.0
        assert config.confidence_threshold == 0.7

    def test_detection_min_snr_validation(self):
        """Test min_snr must be > 0."""
        with pytest.raises(ValueError):
            DetectionConfig(min_snr=0.0)

        with pytest.raises(ValueError):
            DetectionConfig(min_snr=-1.0)

    def test_detection_depth_validation(self):
        """Test transit depth constraints."""
        with pytest.raises(ValueError):
            DetectionConfig(min_transit_depth=1.5)  # > 1.0

        with pytest.raises(ValueError):
            DetectionConfig(min_transit_depth=-0.1)  # < 0.0

    def test_detection_duration_validation(self):
        """Test transit duration must be > 0."""
        with pytest.raises(ValueError):
            DetectionConfig(min_transit_duration_hours=0.0)

    def test_detection_confidence_threshold_validation(self):
        """Test confidence_threshold is in [0, 1]."""
        with pytest.raises(ValueError):
            DetectionConfig(confidence_threshold=1.5)

        with pytest.raises(ValueError):
            DetectionConfig(confidence_threshold=-0.1)


class TestTrainingConfig:
    """Test TrainingConfig validation and functionality."""

    def test_default_training_config(self):
        """Test creating TrainingConfig with defaults."""
        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.learning_rate == 1e-3
        assert config.train_test_split == 0.8
        assert config.validation_split == 0.1
        assert config.random_seed == 42
        assert config.early_stopping_patience == 10

    def test_training_batch_size_validation(self):
        """Test batch_size must be >= 1."""
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)

    def test_training_epochs_validation(self):
        """Test epochs must be >= 1."""
        with pytest.raises(ValueError):
            TrainingConfig(epochs=0)

    def test_training_learning_rate_validation(self):
        """Test learning_rate must be > 0."""
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0.0)

        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-0.001)

    def test_training_split_validation(self):
        """Test train_test_split must be in (0, 1)."""
        with pytest.raises(ValueError):
            TrainingConfig(train_test_split=0.0)

        with pytest.raises(ValueError):
            TrainingConfig(train_test_split=1.0)

    def test_training_validation_split_validation(self):
        """Test validation_split must be in (0, 1)."""
        with pytest.raises(ValueError):
            TrainingConfig(validation_split=0.0)

        with pytest.raises(ValueError):
            TrainingConfig(validation_split=1.0)


class TestProjectConfig:
    """Test top-level ProjectConfig."""

    def test_default_project_config(self):
        """Test creating ProjectConfig with all defaults."""
        config = ProjectConfig()

        assert config.project_name == "exoplanet-discovery"
        assert config.version == "0.1.0"
        assert config.debug is False
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.feature_extraction, FeatureExtractionConfig)
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_project_config_custom_values(self):
        """Test ProjectConfig with custom sub-configurations."""
        config = ProjectConfig(
            debug=True,
            data=DataConfig(max_nan_fraction=0.3),
            training=TrainingConfig(epochs=200),
        )

        assert config.debug is True
        assert config.data.max_nan_fraction == 0.3
        assert config.training.epochs == 200

    def test_project_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ProjectConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["project_name"] == "exoplanet-discovery"
        assert "data" in config_dict
        assert "preprocessing" in config_dict

    def test_project_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "project_name": "test-project",
            "debug": True,
            "data": {"max_nan_fraction": 0.25},
        }

        config = ProjectConfig.from_dict(config_dict)

        assert config.project_name == "test-project"
        assert config.debug is True
        assert config.data.max_nan_fraction == 0.25

    def test_project_config_default_factory(self):
        """Test ProjectConfig.default() factory method."""
        config = ProjectConfig.default()

        assert isinstance(config, ProjectConfig)
        assert config.project_name == "exoplanet-discovery"

    def test_project_config_forbids_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            ProjectConfig(unknown_field="value")


class TestConfigurationYAML:
    """Test YAML loading functionality."""

    def test_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        yaml_path = Path("configs/default_config.yaml")

        if yaml_path.exists():
            config = ProjectConfig.from_yaml(yaml_path)

            assert isinstance(config, ProjectConfig)
            assert config.project_name == "exoplanet-discovery"
            assert config.data.raw_data_dir == Path("data/raw")

    def test_config_from_nonexistent_yaml_raises(self):
        """Test that loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ProjectConfig.from_yaml("nonexistent/config.yaml")

    def test_config_to_yaml_string(self):
        """Test converting config to YAML string."""
        config = ProjectConfig()
        yaml_str = config.to_yaml()

        assert isinstance(yaml_str, str)
        assert "project_name:" in yaml_str
        assert "exoplanet-discovery" in yaml_str
