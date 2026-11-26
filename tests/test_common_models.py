"""Unit tests for data models (LightCurve, TransitEvent, ExoplanetCandidate)."""

from datetime import datetime

import numpy as np
import pytest

from src.common.models import ExoplanetCandidate, LightCurve, TransitEvent


class TestLightCurve:
    """Test suite for LightCurve dataclass."""

    def test_valid_light_curve_creation(self):
        """Test creating a valid light curve."""
        time = np.array([0.0, 1.0, 2.0, 3.0])
        flux = np.array([1.0, 0.99, 1.01, 1.0])

        lc = LightCurve(time=time, flux=flux)

        assert len(lc.time) == 4
        assert len(lc.flux) == 4
        assert lc.duration == 3.0

    def test_light_curve_with_errors(self):
        """Test light curve with flux errors."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, 0.99, 1.01])
        flux_err = np.array([0.01, 0.01, 0.01])

        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)

        assert np.allclose(lc.flux_err, flux_err)

    def test_light_curve_with_metadata(self):
        """Test light curve with additional metadata."""
        time = np.array([0.0, 1.0])
        flux = np.array([1.0, 0.99])
        metadata = {"sector": 1, "camera": 1}

        lc = LightCurve(time=time, flux=flux, metadata=metadata)

        assert lc.metadata["sector"] == 1

    def test_mismatched_time_flux_length(self):
        """Test that mismatched time/flux lengths raise ValueError."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, 0.99])

        with pytest.raises(ValueError, match="time and flux arrays must have equal length"):
            LightCurve(time=time, flux=flux)

    def test_mismatched_flux_err_length(self):
        """Test that flux_err with wrong length raises ValueError."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, 0.99, 1.01])
        flux_err = np.array([0.01, 0.01])  # Wrong length

        with pytest.raises(ValueError, match="flux_err must match flux length"):
            LightCurve(time=time, flux=flux, flux_err=flux_err)

    def test_mean_flux_property(self):
        """Test mean_flux property calculation."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([0.9, 1.0, 1.1])

        lc = LightCurve(time=time, flux=flux)

        assert np.isclose(lc.mean_flux, 1.0)

    def test_std_flux_property(self):
        """Test std_flux property calculation."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([0.9, 1.0, 1.1])

        lc = LightCurve(time=time, flux=flux)

        assert lc.std_flux > 0


class TestTransitEvent:
    """Test suite for TransitEvent dataclass."""

    def test_valid_transit_event_creation(self):
        """Test creating a valid transit event."""
        transit = TransitEvent(
            time_center=100.5,
            duration=2.5,
            depth=0.01,
            snr=7.5,
            confidence=0.95,
        )

        assert transit.time_center == 100.5
        assert transit.duration == 2.5
        assert transit.depth == 0.01
        assert transit.snr == 7.5
        assert transit.is_periodic is False

    def test_transit_event_with_period(self):
        """Test transit event with orbital period."""
        transit = TransitEvent(
            time_center=100.5,
            duration=2.5,
            depth=0.01,
            snr=7.5,
            period=10.0,
        )

        assert transit.is_periodic is True
        assert transit.period == 10.0

    def test_invalid_duration(self):
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration must be positive"):
            TransitEvent(
                time_center=100.0,
                duration=-1.0,
                depth=0.01,
                snr=7.5,
            )

    def test_invalid_depth(self):
        """Test that out-of-range depth raises ValueError."""
        with pytest.raises(ValueError, match="depth must be in"):
            TransitEvent(
                time_center=100.0,
                duration=2.5,
                depth=1.5,  # > 1.0
                snr=7.5,
            )

    def test_invalid_snr(self):
        """Test that negative SNR raises ValueError."""
        with pytest.raises(ValueError, match="snr must be non-negative"):
            TransitEvent(
                time_center=100.0,
                duration=2.5,
                depth=0.01,
                snr=-1.0,
            )

    def test_invalid_confidence(self):
        """Test that out-of-range confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be in"):
            TransitEvent(
                time_center=100.0,
                duration=2.5,
                depth=0.01,
                snr=7.5,
                confidence=1.5,  # > 1.0
            )

    def test_invalid_period(self):
        """Test that non-positive period raises ValueError."""
        with pytest.raises(ValueError, match="period must be positive"):
            TransitEvent(
                time_center=100.0,
                duration=2.5,
                depth=0.01,
                snr=7.5,
                period=-5.0,
            )


class TestExoplanetCandidate:
    """Test suite for ExoplanetCandidate dataclass."""

    def test_valid_candidate_creation(self):
        """Test creating a valid exoplanet candidate."""
        transit = TransitEvent(
            time_center=100.0,
            duration=2.5,
            depth=0.01,
            snr=7.5,
        )

        candidate = ExoplanetCandidate(
            candidate_id="TESS-2024-001",
            target_id="TIC-123456789",
            transit_events=[transit],
            period=10.0,
            radius_earth=1.5,
            equilibrium_temp_k=300.0,
        )

        assert candidate.candidate_id == "TESS-2024-001"
        assert candidate.num_transits == 1
        assert candidate.disposition == "CANDIDATE"

    def test_candidate_with_multiple_transits(self):
        """Test candidate with multiple transit events."""
        transits = [
            TransitEvent(time_center=100.0, duration=2.5, depth=0.01, snr=7.5),
            TransitEvent(time_center=110.0, duration=2.5, depth=0.01, snr=8.0),
            TransitEvent(time_center=120.0, duration=2.5, depth=0.011, snr=7.2),
        ]

        candidate = ExoplanetCandidate(
            candidate_id="TEST-001",
            target_id="TIC-123",
            transit_events=transits,
            period=10.0,
            radius_earth=2.0,
            equilibrium_temp_k=400.0,
        )

        assert candidate.num_transits == 3
        assert candidate.mean_snr == pytest.approx(7.57, abs=0.1)

    def test_candidate_max_confidence(self):
        """Test max_confidence property."""
        transits = [
            TransitEvent(time_center=100.0, duration=2.5, depth=0.01, snr=7.5, confidence=0.85),
            TransitEvent(time_center=110.0, duration=2.5, depth=0.01, snr=8.0, confidence=0.95),
        ]

        candidate = ExoplanetCandidate(
            candidate_id="TEST-002",
            target_id="TIC-456",
            transit_events=transits,
            period=10.0,
            radius_earth=1.0,
            equilibrium_temp_k=350.0,
        )

        assert candidate.max_confidence == 0.95

    def test_empty_transits_rejected(self):
        """Test that empty transit_events list is rejected."""
        with pytest.raises(ValueError, match="transit_events cannot be empty"):
            ExoplanetCandidate(
                candidate_id="TEST-003",
                target_id="TIC-789",
                transit_events=[],
                period=10.0,
                radius_earth=1.0,
                equilibrium_temp_k=350.0,
            )

    def test_invalid_period(self):
        """Test that non-positive period is rejected."""
        transit = TransitEvent(
            time_center=100.0,
            duration=2.5,
            depth=0.01,
            snr=7.5,
        )

        with pytest.raises(ValueError, match="period must be positive"):
            ExoplanetCandidate(
                candidate_id="TEST-004",
                target_id="TIC-000",
                transit_events=[transit],
                period=-5.0,  # Invalid
                radius_earth=1.0,
                equilibrium_temp_k=350.0,
            )

    def test_invalid_radius(self):
        """Test that negative radius is rejected."""
        transit = TransitEvent(
            time_center=100.0,
            duration=2.5,
            depth=0.01,
            snr=7.5,
        )

        with pytest.raises(ValueError, match="radius_earth must be non-negative"):
            ExoplanetCandidate(
                candidate_id="TEST-005",
                target_id="TIC-111",
                transit_events=[transit],
                period=10.0,
                radius_earth=-1.0,  # Invalid
                equilibrium_temp_k=350.0,
            )

    def test_invalid_temperature(self):
        """Test that negative temperature is rejected."""
        transit = TransitEvent(
            time_center=100.0,
            duration=2.5,
            depth=0.01,
            snr=7.5,
        )

        with pytest.raises(ValueError, match="equilibrium_temp_k must be non-negative"):
            ExoplanetCandidate(
                candidate_id="TEST-006",
                target_id="TIC-222",
                transit_events=[transit],
                period=10.0,
                radius_earth=1.0,
                equilibrium_temp_k=-100.0,  # Invalid
            )

    def test_candidate_custom_disposition(self):
        """Test candidate with custom disposition."""
        transit = TransitEvent(
            time_center=100.0,
            duration=2.5,
            depth=0.01,
            snr=7.5,
        )

        candidate = ExoplanetCandidate(
            candidate_id="TEST-007",
            target_id="TIC-333",
            transit_events=[transit],
            period=10.0,
            radius_earth=1.0,
            equilibrium_temp_k=350.0,
            disposition="CONFIRMED",
        )

        assert candidate.disposition == "CONFIRMED"

    def test_candidate_discovered_at_timestamp(self):
        """Test that discovered_at timestamp is set."""
        transit = TransitEvent(
            time_center=100.0,
            duration=2.5,
            depth=0.01,
            snr=7.5,
        )

        before = datetime.utcnow()
        candidate = ExoplanetCandidate(
            candidate_id="TEST-008",
            target_id="TIC-444",
            transit_events=[transit],
            period=10.0,
            radius_earth=1.0,
            equilibrium_temp_k=350.0,
        )
        after = datetime.utcnow()

        assert before <= candidate.discovered_at <= after
