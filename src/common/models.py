"""
Strongly typed data models for exoplanet discovery and light-curve analysis.

This module defines the core data structures used throughout the project,
including light curves, transit events, and exoplanet candidates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class LightCurve:
    """
    Represents a normalized light curve (flux measurements over time).

    Attributes:
        time: Array of time values (days, typically).
        flux: Array of normalized flux measurements (unitless, ~1.0 nominal).
        flux_err: Optional array of flux measurement uncertainties.
        sector: Observation sector or mission identifier.
        target_id: Unique identifier for the observed target (star/system).
        metadata: Optional dictionary of additional observational metadata.

    Raises:
        ValueError: If time and flux arrays have mismatched lengths or if
            flux_err is provided but has incorrect shape.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None
    sector: str | None = None
    target_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate light curve data integrity."""
        if len(self.time) != len(self.flux):
            raise ValueError(
                f"time and flux arrays must have equal length. "
                f"Got len(time)={len(self.time)}, len(flux)={len(self.flux)}"
            )
        if self.flux_err is not None and len(self.flux_err) != len(self.flux):
            raise ValueError(
                f"flux_err must match flux length. "
                f"Got len(flux_err)={len(self.flux_err)}, len(flux)={len(self.flux)}"
            )

    @property
    def duration(self) -> float:
        """Total observation duration in time units."""
        return float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0

    @property
    def mean_flux(self) -> float:
        """Mean normalized flux value."""
        return float(np.mean(self.flux))

    @property
    def std_flux(self) -> float:
        """Standard deviation of flux values."""
        return float(np.std(self.flux))


@dataclass
class TransitEvent:
    """
    Represents a single transit event (planet blocking star light).

    Attributes:
        time_center: Time of transit center (days).
        duration: Transit duration (hours).
        depth: Transit depth as fractional change in flux (e.g., 0.01 = 1%).
        period: Orbital period if known (days). None indicates single transit.
        snr: Signal-to-noise ratio of the detection.
        confidence: Confidence score in [0, 1] for anomaly/transit classification.
        metadata: Optional dictionary of additional event metadata.

    Raises:
        ValueError: If duration <= 0, depth < 0 or > 1, snr < 0, or
            confidence not in [0, 1].
    """

    time_center: float
    duration: float
    depth: float
    snr: float
    confidence: float = 0.5
    period: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate transit event parameters."""
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if not 0 <= self.depth <= 1:
            raise ValueError(f"depth must be in [0, 1], got {self.depth}")
        if self.snr < 0:
            raise ValueError(f"snr must be non-negative, got {self.snr}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.period is not None and self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def is_periodic(self) -> bool:
        """Whether the transit has an associated orbital period."""
        return self.period is not None


@dataclass
class ExoplanetCandidate:
    """
    Represents an exoplanet discovery candidate with detection metadata.

    Attributes:
        candidate_id: Unique identifier for this candidate.
        target_id: Identifier of the host star.
        transit_events: List of observed transit events for this candidate.
        period: Orbital period (days).
        radius_earth: Estimated planetary radius in Earth radii.
        equilibrium_temp_k: Estimated equilibrium temperature (Kelvin).
        disposition: Classification status (e.g., "CANDIDATE", "CONFIRMED", "FALSE_POSITIVE").
        discovered_at: Timestamp of discovery/creation.
        source: Data source identifier (e.g., "TESS", "Kepler").
        notes: Optional analysis notes and metadata.

    Raises:
        ValueError: If period <= 0, radius_earth < 0, equilibrium_temp_k < 0,
            or transit_events is empty.
    """

    candidate_id: str
    target_id: str
    transit_events: list[TransitEvent]
    period: float
    radius_earth: float
    equilibrium_temp_k: float
    disposition: str = "CANDIDATE"
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    source: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate exoplanet candidate parameters."""
        if not self.transit_events:
            raise ValueError("transit_events cannot be empty")
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.radius_earth < 0:
            raise ValueError(f"radius_earth must be non-negative, got {self.radius_earth}")
        if self.equilibrium_temp_k < 0:
            raise ValueError(
                f"equilibrium_temp_k must be non-negative, got {self.equilibrium_temp_k}"
            )

    @property
    def num_transits(self) -> int:
        """Number of observed transit events."""
        return len(self.transit_events)

    @property
    def mean_snr(self) -> float:
        """Mean SNR across all transit events."""
        return float(np.mean([t.snr for t in self.transit_events]))

    @property
    def max_confidence(self) -> float:
        """Maximum confidence score across all transit events."""
        return float(np.max([t.confidence for t in self.transit_events]))
