"""Strongly typed data models for exoplanet discovery.

This module defines the core data structures used throughout the pipeline:
- LightCurve: Photometric time-series data from space telescopes
- TransitEvent: A detected transit event in a light curve
- ExoplanetCandidate: A candidate exoplanet with associated metadata
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class LightCurve:
    """Represents a photometric light curve from a space telescope.

    Attributes:
        time: Time measurements in days (relative to mission start or absolute).
        flux: Normalized flux measurements (dimensionless, typically 0-2 range).
        flux_err: Measurement uncertainties (standard deviation per point).
        tic_id: TESS Input Catalog ID (if TESS data) or KIC ID (if Kepler).
        sector: Observing sector/quarter (metadata).
        source: Data source ('Kepler', 'TESS', 'synthetic', etc.).
        metadata: Additional metadata (e.g., instrument, cadence, RA/Dec).

    Raises:
        ValueError: If time and flux lengths don't match, or contain NaN/inf.
    """

    time: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]
    flux_err: npt.NDArray[np.float64]
    tic_id: str
    sector: int | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate light curve data integrity."""
        if len(self.time) != len(self.flux):
            raise ValueError(
                f"time and flux length mismatch: {len(self.time)} vs {len(self.flux)}"
            )
        if len(self.time) != len(self.flux_err):
            raise ValueError(
                f"time and flux_err length mismatch: "
                f"{len(self.time)} vs {len(self.flux_err)}"
            )
        if np.any(~np.isfinite(self.time)):
            raise ValueError("time contains NaN or inf values")
        if np.any(~np.isfinite(self.flux)):
            raise ValueError("flux contains NaN or inf values")
        if np.any(~np.isfinite(self.flux_err)):
            raise ValueError("flux_err contains NaN or inf values")
        if np.any(self.flux_err <= 0):
            raise ValueError("flux_err must be positive")
        if len(self.time) < 10:
            raise ValueError("light curve must have at least 10 points")

    def __len__(self) -> int:
        """Return the number of cadences."""
        return len(self.time)

    @property
    def duration_days(self) -> float:
        """Total observation duration in days."""
        return float(self.time[-1] - self.time[0])

    @property
    def signal_to_noise_estimate(self) -> float:
        """Rough SNR estimate: std(flux) / median(flux_err)."""
        flux_std = float(np.std(self.flux))
        err_median = float(np.median(self.flux_err))
        return flux_std / (err_median + 1e-10)


@dataclass
class TransitEvent:
    """Represents a detected or suspected transit event.

    Attributes:
        transit_time: Time of transit center (or mid-transit time).
        transit_depth_ppm: Transit depth in parts per million (ppm).
        transit_depth_err: Uncertainty on depth (ppm).
        duration_hours: Transit duration in hours.
        period_days: Orbital period in days (if periodic signal detected).
        snr: Signal-to-noise ratio of the detection.
        confidence: Detection confidence score [0, 1].
    """

    transit_time: float
    transit_depth_ppm: float
    transit_depth_err: float
    duration_hours: float
    period_days: float | None = None
    snr: float = 0.0
    confidence: float = 0.5

    def __post_init__(self) -> None:
        """Validate transit event parameters."""
        if self.transit_depth_ppm <= 0:
            raise ValueError("transit_depth_ppm must be positive")
        if self.transit_depth_err < 0:
            raise ValueError("transit_depth_err must be non-negative")
        if self.duration_hours <= 0:
            raise ValueError("duration_hours must be positive")
        if self.period_days is not None and self.period_days <= 0:
            raise ValueError("period_days must be positive")
        if not 0 <= self.snr:
            raise ValueError("snr must be non-negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class ExoplanetCandidate:
    """Represents a candidate exoplanet derived from light curve analysis.

    Attributes:
        tic_id: TESS/Kepler object identifier.
        candidate_name: Human-readable candidate name (e.g., 'TOI-1234 b').
        exoplanet_probability: ML-model probability [0, 1].
        transit_events: List of associated transit events.
        estimated_radius_earth: Estimated planet radius in Earth radii.
        estimated_period_days: Estimated orbital period in days.
        host_star_mag: Host star apparent magnitude (V-band).
        notes: Additional notes or flags (e.g., eclipsing binary, false positive).
    """

    tic_id: str
    candidate_name: str
    exoplanet_probability: float
    transit_events: list[TransitEvent] = field(default_factory=list)
    estimated_radius_earth: float | None = None
    estimated_period_days: float | None = None
    host_star_mag: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate candidate parameters."""
        if not 0 <= self.exoplanet_probability <= 1:
            raise ValueError(
                f"exoplanet_probability must be in [0, 1], "
                f"got {self.exoplanet_probability}"
            )
        if self.estimated_radius_earth is not None:
            if self.estimated_radius_earth <= 0:
                raise ValueError("estimated_radius_earth must be positive")
        if self.estimated_period_days is not None:
            if self.estimated_period_days <= 0:
                raise ValueError("estimated_period_days must be positive")

    def is_likely_exoplanet(self, threshold: float = 0.5) -> bool:
        """Check if candidate exceeds probability threshold.

        Args:
            threshold: Probability threshold [0, 1].

        Returns:
            True if exoplanet_probability >= threshold.
        """
        return self.exoplanet_probability >= threshold

    @property
    def num_transits(self) -> int:
        """Number of associated transit events."""
        return len(self.transit_events)

    @property
    def best_transit(self) -> TransitEvent | None:
        """Return transit with highest SNR, if any."""
        if not self.transit_events:
            return None
        return max(self.transit_events, key=lambda t: t.snr)
