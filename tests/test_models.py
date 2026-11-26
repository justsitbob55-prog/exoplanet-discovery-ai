"""Unit tests for exoplanet_discovery.common.models.

Tests cover:
- LightCurve: validation, properties, edge cases
- TransitEvent: validation, properties
- ExoplanetCandidate: validation, methods
"""

import numpy as np
import pytest

from exoplanet_discovery.common.models import (
    ExoplanetCandidate,
    LightCurve,
    TransitEvent,
)


class TestLightCurve:
    """Tests for LightCurve model."""

    def test_light_curve_creation_valid(self) -> None:
        """Test creating a valid light curve."""
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        flux = np.array([1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 1.01, 0.99, 1.0, 1.01])
        flux_err = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

        lc = LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            tic_id="TIC 123456789",
            sector=1,
            source="TESS",
        )

        assert lc.tic_id == "TIC 123456789"
        assert len(lc) == 10
        assert lc.source == "TESS"

    def test_light_curve_length_mismatch_time_flux(self) -> None:
        """Test validation: time and flux length mismatch."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, 1.01, 0.99, 1.0])
        flux_err = np.array([0.001, 0.001, 0.001, 0.001])

        with pytest.raises(ValueError, match="time and flux length mismatch"):
            LightCurve(
                time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123"
            )

    def test_light_curve_length_mismatch_time_flux_err(self) -> None:
        """Test validation: time and flux_err length mismatch."""
        time = np.array([0.0, 1.0, 2.0, 3.0])
        flux = np.array([1.0, 1.01, 0.99, 1.0])
        flux_err = np.array([0.001, 0.001, 0.001])

        with pytest.raises(ValueError, match="time and flux_err length mismatch"):
            LightCurve(
                time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123"
            )

    def test_light_curve_nan_in_time(self) -> None:
        """Test validation: NaN in time array."""
        time = np.array([0.0, np.nan, 2.0])
        flux = np.array([1.0, 1.01, 0.99])
        flux_err = np.array([0.001, 0.001, 0.001])

        with pytest.raises(ValueError, match="time contains NaN"):
            LightCurve(time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123")

    def test_light_curve_inf_in_flux(self) -> None:
        """Test validation: inf in flux array."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, np.inf, 0.99])
        flux_err = np.array([0.001, 0.001, 0.001])

        with pytest.raises(ValueError, match="flux contains NaN"):
            LightCurve(time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123")

    def test_light_curve_negative_flux_err(self) -> None:
        """Test validation: negative flux_err."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, 1.01, 0.99])
        flux_err = np.array([0.001, -0.001, 0.001])

        with pytest.raises(ValueError, match="flux_err must be positive"):
            LightCurve(time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123")

    def test_light_curve_too_few_cadences(self) -> None:
        """Test validation: insufficient cadences."""
        time = np.array([0.0, 1.0])
        flux = np.array([1.0, 1.01])
        flux_err = np.array([0.001, 0.001])

        with pytest.raises(ValueError, match="at least 10 points"):
            LightCurve(time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123")

    def test_light_curve_duration_days(self) -> None:
        """Test duration_days property."""
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        flux = np.ones(10)
        flux_err = np.ones(10) * 0.001

        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123")
        assert lc.duration_days == 9.0

    def test_light_curve_signal_to_noise_estimate(self) -> None:
        """Test SNR estimation."""
        time = np.linspace(0, 10, 100)
        flux = np.ones(100) + np.random.normal(0, 0.01, 100)
        flux_err = np.ones(100) * 0.001

        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123")
        snr = lc.signal_to_noise_estimate
        assert snr > 0

    def test_light_curve_frozen(self) -> None:
        """Test that LightCurve is frozen (immutable)."""
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        flux = np.ones(10)
        flux_err = np.ones(10) * 0.001

        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, tic_id="TIC 123")

        with pytest.raises(AttributeError):
            lc.tic_id = "NEW_ID"  # type: ignore


class TestTransitEvent:
    """Tests for TransitEvent model."""

    def test_transit_event_creation_valid(self) -> None:
        """Test creating a valid transit event."""
        te = TransitEvent(
            transit_time=100.5,
            transit_depth_ppm=500.0,
            transit_depth_err=50.0,
            duration_hours=2.5,
            period_days=10.0,
            snr=7.5,
            confidence=0.85,
        )

        assert te.transit_time == 100.5
        assert te.transit_depth_ppm == 500.0
        assert te.confidence == 0.85

    def test_transit_event_negative_depth(self) -> None:
        """Test validation: negative transit depth."""
        with pytest.raises(ValueError, match="transit_depth_ppm must be positive"):
            TransitEvent(
                transit_time=100.5,
                transit_depth_ppm=-500.0,
                transit_depth_err=50.0,
                duration_hours=2.5,
            )

    def test_transit_event_negative_duration(self) -> None:
        """Test validation: negative duration."""
        with pytest.raises(ValueError, match="duration_hours must be positive"):
            TransitEvent(
                transit_time=100.5,
                transit_depth_ppm=500.0,
                transit_depth_err=50.0,
                duration_hours=-2.5,
            )

    def test_transit_event_negative_period(self) -> None:
        """Test validation: negative period."""
        with pytest.raises(ValueError, match="period_days must be positive"):
            TransitEvent(
                transit_time=100.5,
                transit_depth_ppm=500.0,
                transit_depth_err=50.0,
                duration_hours=2.5,
                period_days=-10.0,
            )

    def test_transit_event_confidence_out_of_range(self) -> None:
        """Test validation: confidence out of [0, 1]."""
        with pytest.raises(ValueError, match="confidence must be in"):
            TransitEvent(
                transit_time=100.5,
                transit_depth_ppm=500.0,
                transit_depth_err=50.0,
                duration_hours=2.5,
                confidence=1.5,
            )

    def test_transit_event_negative_snr(self) -> None:
        """Test validation: negative SNR."""
        with pytest.raises(ValueError, match="snr must be non-negative"):
            TransitEvent(
                transit_time=100.5,
                transit_depth_ppm=500.0,
                transit_depth_err=50.0,
                duration_hours=2.5,
                snr=-1.0,
            )


class TestExoplanetCandidate:
    """Tests for ExoplanetCandidate model."""

    def test_candidate_creation_valid(self) -> None:
        """Test creating a valid exoplanet candidate."""
        transit = TransitEvent(
            transit_time=100.5,
            transit_depth_ppm=500.0,
            transit_depth_err=50.0,
            duration_hours=2.5,
            snr=7.5,
            confidence=0.85,
        )

        candidate = ExoplanetCandidate(
            tic_id="TIC 123456789",
            candidate_name="TOI-1234 b",
            exoplanet_probability=0.92,
            transit_events=[transit],
            estimated_radius_earth=2.5,
            estimated_period_days=10.0,
            host_star_mag=12.5,
        )

        assert candidate.tic_id == "TIC 123456789"
        assert candidate.exoplanet_probability == 0.92
        assert candidate.num_transits == 1

    def test_candidate_probability_out_of_range(self) -> None:
        """Test validation: probability out of [0, 1]."""
        with pytest.raises(ValueError, match="exoplanet_probability must be in"):
            ExoplanetCandidate(
                tic_id="TIC 123",
                candidate_name="TOI-123 b",
                exoplanet_probability=1.5,
            )

    def test_candidate_negative_radius(self) -> None:
        """Test validation: negative estimated radius."""
        with pytest.raises(ValueError, match="estimated_radius_earth must be positive"):
            ExoplanetCandidate(
                tic_id="TIC 123",
                candidate_name="TOI-123 b",
                exoplanet_probability=0.8,
                estimated_radius_earth=-2.5,
            )

    def test_candidate_negative_period(self) -> None:
        """Test validation: negative estimated period."""
        with pytest.raises(ValueError, match="estimated_period_days must be positive"):
            ExoplanetCandidate(
                tic_id="TIC 123",
                candidate_name="TOI-123 b",
                exoplanet_probability=0.8,
                estimated_period_days=-10.0,
            )

    def test_candidate_is_likely_exoplanet(self) -> None:
        """Test is_likely_exoplanet method."""
        candidate = ExoplanetCandidate(
            tic_id="TIC 123",
            candidate_name="TOI-123 b",
            exoplanet_probability=0.92,
        )

        assert candidate.is_likely_exoplanet(threshold=0.5)
        assert candidate.is_likely_exoplanet(threshold=0.9)
        assert not candidate.is_likely_exoplanet(threshold=0.95)

    def test_candidate_num_transits(self) -> None:
        """Test num_transits property."""
        transits = [
            TransitEvent(
                transit_time=100.0,
                transit_depth_ppm=500.0,
                transit_depth_err=50.0,
                duration_hours=2.5,
            ),
            TransitEvent(
                transit_time=110.0,
                transit_depth_ppm=480.0,
                transit_depth_err=50.0,
                duration_hours=2.5,
            ),
        ]

        candidate = ExoplanetCandidate(
            tic_id="TIC 123",
            candidate_name="TOI-123 b",
            exoplanet_probability=0.92,
            transit_events=transits,
        )

        assert candidate.num_transits == 2

    def test_candidate_best_transit(self) -> None:
        """Test best_transit property."""
        transits = [
            TransitEvent(
                transit_time=100.0,
                transit_depth_ppm=500.0,
                transit_depth_err=50.0,
                duration_hours=2.5,
                snr=5.0,
            ),
            TransitEvent(
                transit_time=110.0,
                transit_depth_ppm=480.0,
                transit_depth_err=50.0,
                duration_hours=2.5,
                snr=10.0,
            ),
        ]

        candidate = ExoplanetCandidate(
            tic_id="TIC 123",
            candidate_name="TOI-123 b",
            exoplanet_probability=0.92,
            transit_events=transits,
        )

        best = candidate.best_transit
        assert best is not None
        assert best.snr == 10.0

    def test_candidate_best_transit_empty(self) -> None:
        """Test best_transit with no events."""
        candidate = ExoplanetCandidate(
            tic_id="TIC 123",
            candidate_name="TOI-123 b",
            exoplanet_probability=0.92,
        )

        assert candidate.best_transit is None
