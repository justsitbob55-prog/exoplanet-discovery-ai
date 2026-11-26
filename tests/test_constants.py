"""Unit tests for exoplanet_discovery.common.constants.

Tests verify that:
- All constants are defined and have correct types
- Physical constants have reasonable values
- Threshold values are in valid ranges
- No constants have been accidentally changed
"""


from exoplanet_discovery.common import constants


class TestPhysicalConstants:
    """Tests for physical and astronomical constants."""

    def test_earth_radius_reasonable(self) -> None:
        """Earth radius should be positive and roughly correct."""
        assert constants.EARTH_RADIUS_KM > 0
        assert 6300 < constants.EARTH_RADIUS_KM < 6500

    def test_earth_mass_reasonable(self) -> None:
        """Earth mass should be positive and roughly correct."""
        assert constants.EARTH_MASS_KG > 0
        assert 5e24 < constants.EARTH_MASS_KG < 6e24

    def test_solar_radius_reasonable(self) -> None:
        """Solar radius should be much larger than Earth radius."""
        assert constants.SOLAR_RADIUS_KM > constants.EARTH_RADIUS_KM
        assert 690000 < constants.SOLAR_RADIUS_KM < 700000

    def test_solar_mass_reasonable(self) -> None:
        """Solar mass should be much larger than Earth mass."""
        assert constants.SOLAR_MASS_KG > constants.EARTH_MASS_KG
        assert 1.9e30 < constants.SOLAR_MASS_KG < 2e30

    def test_solar_mass_to_earth_mass_ratio(self) -> None:
        """Check Sun/Earth mass ratio (should be ~330,000)."""
        ratio = constants.SOLAR_MASS_KG / constants.EARTH_MASS_KG
        assert 320000 < ratio < 340000


class TestTransitDetectionConstants:
    """Tests for transit detection constants."""

    def test_minimum_transit_depth_positive(self) -> None:
        """Minimum transit depth must be positive."""
        assert constants.MINIMUM_TRANSIT_DEPTH_PPM > 0

    def test_minimum_transit_duration_reasonable(self) -> None:
        """Minimum transit duration should be between 10 and 60 minutes."""
        assert 10 <= constants.MINIMUM_TRANSIT_DURATION_MINUTES <= 60

    def test_maximum_transit_duration_reasonable(self) -> None:
        """Maximum transit duration should be positive and less than a day."""
        assert 0 < constants.MAXIMUM_TRANSIT_DURATION_HOURS < 24

    def test_transit_duration_range_consistent(self) -> None:
        """Max duration should be greater than min duration."""
        min_hours = constants.MINIMUM_TRANSIT_DURATION_MINUTES / 60
        assert constants.MAXIMUM_TRANSIT_DURATION_HOURS > min_hours


class TestLightCurveConstants:
    """Tests for light curve handling constants."""

    def test_minimum_cadences_reasonable(self) -> None:
        """Minimum number of cadences should be reasonable."""
        assert 5 <= constants.MINIMUM_LIGHT_CURVE_CADENCES <= 100

    def test_maximum_missing_cadence_fraction_in_range(self) -> None:
        """Maximum missing cadence fraction should be in [0, 1]."""
        assert 0 <= constants.MAXIMUM_MISSING_CADENCE_FRACTION <= 1


class TestPreprocessingConstants:
    """Tests for preprocessing constants."""

    def test_default_detrend_window_positive(self) -> None:
        """Detrend window size must be positive."""
        assert constants.DEFAULT_DETREND_WINDOW_SIZE > 0

    def test_default_detrend_poly_order_reasonable(self) -> None:
        """Polynomial order should be between 1 and 5."""
        assert 1 <= constants.DEFAULT_DETREND_POLY_ORDER <= 5

    def test_normalization_method_valid_string(self) -> None:
        """Normalization method should be non-empty string."""
        assert isinstance(constants.DEFAULT_FLUX_NORMALIZATION_METHOD, str)
        assert len(constants.DEFAULT_FLUX_NORMALIZATION_METHOD) > 0


class TestFeatureExtractionConstants:
    """Tests for feature extraction constants."""

    def test_minimum_period_positive(self) -> None:
        """Minimum period must be positive."""
        assert constants.MINIMUM_PERIOD_DAYS > 0

    def test_maximum_period_positive(self) -> None:
        """Maximum period must be positive."""
        assert constants.MAXIMUM_PERIOD_DAYS > 0

    def test_period_range_consistent(self) -> None:
        """Maximum period should be greater than minimum."""
        assert constants.MAXIMUM_PERIOD_DAYS > constants.MINIMUM_PERIOD_DAYS

    def test_frequency_resolution_positive(self) -> None:
        """Frequency resolution must be positive."""
        assert constants.DEFAULT_FREQUENCY_RESOLUTION_UHERTZ > 0


class TestDetectionConstants:
    """Tests for detection/classification constants."""

    def test_default_confidence_threshold_in_range(self) -> None:
        """Confidence threshold should be in [0, 1]."""
        assert 0 <= constants.DEFAULT_EXOPLANET_CONFIDENCE_THRESHOLD <= 1

    def test_high_confidence_threshold_in_range(self) -> None:
        """High confidence threshold should be in [0, 1]."""
        assert 0 <= constants.HIGH_CONFIDENCE_THRESHOLD <= 1

    def test_low_confidence_threshold_in_range(self) -> None:
        """Low confidence threshold should be in [0, 1]."""
        assert 0 <= constants.LOW_CONFIDENCE_THRESHOLD <= 1

    def test_confidence_thresholds_ordered(self) -> None:
        """Confidence thresholds should be ordered."""
        assert (
            constants.LOW_CONFIDENCE_THRESHOLD
            < constants.DEFAULT_EXOPLANET_CONFIDENCE_THRESHOLD
            < constants.HIGH_CONFIDENCE_THRESHOLD
        )

    def test_minimum_snr_positive(self) -> None:
        """Minimum SNR must be positive."""
        assert constants.MINIMUM_SNR_FOR_DETECTION > 0


class TestProcessingConstants:
    """Tests for data processing constants."""

    def test_default_batch_size_positive(self) -> None:
        """Batch size must be positive."""
        assert constants.DEFAULT_BATCH_SIZE > 0

    def test_random_seed_integer(self) -> None:
        """Random seed should be an integer."""
        assert isinstance(constants.RANDOM_SEED, int)
        assert constants.RANDOM_SEED >= 0


class TestApplicationConstants:
    """Tests for application-level constants."""

    def test_default_log_level_valid(self) -> None:
        """Log level should be a valid string."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert constants.DEFAULT_LOG_LEVEL in valid_levels

    def test_max_upload_size_positive(self) -> None:
        """Max upload size must be positive."""
        assert constants.DEFAULT_MAX_UPLOAD_SIZE_MB > 0


class TestFileConstants:
    """Tests for file-related constants."""

    def test_light_curve_file_extensions_not_empty(self) -> None:
        """File extensions tuple should not be empty."""
        assert len(constants.LIGHT_CURVE_FILE_EXTENSIONS) > 0

    def test_light_curve_file_extensions_all_strings(self) -> None:
        """All extensions should be strings starting with dot."""
        for ext in constants.LIGHT_CURVE_FILE_EXTENSIONS:
            assert isinstance(ext, str)
            assert ext.startswith(".")

    def test_model_checkpoint_extension_valid(self) -> None:
        """Model checkpoint extension should be a valid string."""
        assert isinstance(constants.MODEL_CHECKPOINT_EXTENSION, str)
        assert constants.MODEL_CHECKPOINT_EXTENSION.startswith(".")

    def test_config_file_path_valid(self) -> None:
        """Default config file path should be a valid string."""
        assert isinstance(constants.DEFAULT_CONFIG_FILE, str)
        assert len(constants.DEFAULT_CONFIG_FILE) > 0
