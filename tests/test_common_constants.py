"""Unit tests for project constants."""

import pytest

from src.common import constants


class TestAstronomicalConstants:
    """Test astronomical reference constants."""

    def test_earth_radius_values(self):
        """Test Earth radius constants are reasonable."""
        assert constants.EARTH_RADIUS_KM == 6371.0
        assert constants.EARTH_RADIUS_M == pytest.approx(6371000.0)
        assert constants.EARTH_RADII_TO_KM == 6371.0

    def test_solar_radius_values(self):
        """Test solar radius constants are reasonable."""
        assert constants.SOLAR_RADIUS_KM == 696000.0
        assert constants.SOLAR_RADIUS_M == pytest.approx(696000000.0)
        assert constants.SOLAR_RADII_TO_KM == 696000.0

    def test_jupiter_radius_values(self):
        """Test Jupiter radius constants are reasonable."""
        assert constants.JUPITER_RADIUS_KM == 69911.0
        assert constants.JUPITER_RADII_TO_KM == 69911.0

    def test_time_conversion_constants(self):
        """Test time conversion constants."""
        assert constants.SECONDS_PER_DAY == 86400.0
        assert constants.HOURS_PER_DAY == 24.0
        assert constants.MINUTES_PER_HOUR == 60.0


class TestLightCurveConstraints:
    """Test light curve validation constraints."""

    def test_flux_range_constants(self):
        """Test flux value range constants."""
        assert constants.MIN_VALID_FLUX == 0.0
        assert constants.MAX_VALID_FLUX == 2.0

    def test_transit_depth_range(self):
        """Test transit depth constraints."""
        assert constants.MIN_TRANSIT_DEPTH == 0.0001
        assert constants.MAX_TRANSIT_DEPTH == 1.0

    def test_transit_duration_range(self):
        """Test transit duration constraints."""
        assert constants.MIN_TRANSIT_DURATION_HOURS == 0.5
        assert constants.MAX_TRANSIT_DURATION_HOURS == 100.0

    def test_snr_thresholds(self):
        """Test signal-to-noise ratio thresholds."""
        assert constants.MIN_SNR_FOR_DETECTION == 3.0
        assert constants.STRONG_SNR_THRESHOLD == 7.0

    def test_confidence_bounds(self):
        """Test confidence score bounds."""
        assert constants.MIN_CONFIDENCE == 0.0
        assert constants.MAX_CONFIDENCE == 1.0


class TestProcessingParameters:
    """Test data processing parameters."""

    def test_normalization_parameters(self):
        """Test flux normalization parameters."""
        assert constants.FLUX_NORMALIZATION_PERCENTILE == 95.0
        assert constants.SIGMA_CLIPPING_THRESHOLD == 5.0

    def test_detrending_parameters(self):
        """Test detrending parameters."""
        assert constants.POLYNOMIAL_DETREND_ORDER == 3
        assert constants.SAVGOL_WINDOW_LENGTH == 51

    def test_feature_extraction_parameters(self):
        """Test feature extraction parameters."""
        assert constants.N_FEATURES_BASE == 32
        assert constants.N_WAVELETS == 8

    def test_model_parameters(self):
        """Test model architecture parameters."""
        assert constants.MODEL_EMBEDDING_DIM == 128
        assert constants.MODEL_HIDDEN_DIM == 256

    def test_training_defaults(self):
        """Test training default parameters."""
        assert constants.DEFAULT_BATCH_SIZE == 32
        assert constants.DEFAULT_EPOCHS == 100
        assert constants.DEFAULT_LEARNING_RATE == 1e-3
        assert constants.DEFAULT_TRAIN_TEST_SPLIT == 0.8


class TestFileFormats:
    """Test supported file format constants."""

    def test_light_curve_formats(self):
        """Test supported light curve data formats."""
        assert "fits" in constants.SUPPORTED_LIGHT_CURVE_FORMATS
        assert "csv" in constants.SUPPORTED_LIGHT_CURVE_FORMATS
        assert "hdf5" in constants.SUPPORTED_LIGHT_CURVE_FORMATS
        assert "parquet" in constants.SUPPORTED_LIGHT_CURVE_FORMATS

    def test_config_formats(self):
        """Test supported configuration formats."""
        assert "yaml" in constants.SUPPORTED_CONFIG_FORMATS
        assert "json" in constants.SUPPORTED_CONFIG_FORMATS

    def test_file_extensions(self):
        """Test file extension constants."""
        assert constants.FITS_EXTENSION == ".fits"
        assert constants.CSV_EXTENSION == ".csv"
        assert constants.HDF5_EXTENSION == ".h5"
        assert constants.PARQUET_EXTENSION == ".parquet"
        assert constants.YAML_EXTENSION == ".yaml"
        assert constants.JSON_EXTENSION == ".json"


class TestMissions:
    """Test space mission constants."""

    def test_mission_names(self):
        """Test mission identifier constants."""
        assert constants.MISSION_TESS == "TESS"
        assert constants.MISSION_KEPLER == "Kepler"
        assert constants.MISSION_K2 == "K2"

    def test_supported_missions_list(self):
        """Test that all missions are in supported list."""
        assert constants.MISSION_TESS in constants.SUPPORTED_MISSIONS
        assert constants.MISSION_KEPLER in constants.SUPPORTED_MISSIONS
        assert constants.MISSION_K2 in constants.SUPPORTED_MISSIONS


class TestDispositions:
    """Test candidate disposition constants."""

    def test_disposition_values(self):
        """Test disposition keyword constants."""
        assert constants.DISPOSITION_CANDIDATE == "CANDIDATE"
        assert constants.DISPOSITION_CONFIRMED == "CONFIRMED"
        assert constants.DISPOSITION_FALSE_POSITIVE == "FALSE_POSITIVE"
        assert constants.DISPOSITION_PLANETARY == "PLANETARY"
        assert constants.DISPOSITION_STELLAR == "STELLAR"
        assert constants.DISPOSITION_INSTRUMENTAL == "INSTRUMENTAL"

    def test_valid_dispositions_list(self):
        """Test that all dispositions are in valid list."""
        assert constants.DISPOSITION_CANDIDATE in constants.VALID_DISPOSITIONS
        assert constants.DISPOSITION_CONFIRMED in constants.VALID_DISPOSITIONS
        assert constants.DISPOSITION_FALSE_POSITIVE in constants.VALID_DISPOSITIONS
        assert constants.DISPOSITION_PLANETARY in constants.VALID_DISPOSITIONS
        assert constants.DISPOSITION_STELLAR in constants.VALID_DISPOSITIONS
        assert constants.DISPOSITION_INSTRUMENTAL in constants.VALID_DISPOSITIONS

    def test_confidence_thresholds(self):
        """Test confidence classification thresholds."""
        assert constants.CONFIDENCE_THRESHOLD_HIGH == 0.9
        assert constants.CONFIDENCE_THRESHOLD_MEDIUM == 0.7
        assert constants.CONFIDENCE_THRESHOLD_LOW == 0.5
        # Verify threshold ordering
        assert (
            constants.CONFIDENCE_THRESHOLD_LOW
            < constants.CONFIDENCE_THRESHOLD_MEDIUM
            < constants.CONFIDENCE_THRESHOLD_HIGH
        )
