"""Cadence-preserving binning for computationally bounded analysis."""

from __future__ import annotations

import numpy as np

from ..common.models import LightCurve


def downsample_light_curve(curve: LightCurve, max_points: int) -> LightCurve:
    """Bin adjacent cadences, preserving mean transit depth and uncertainties."""
    if max_points < 10:
        raise ValueError("max_points must be at least 10.")
    if len(curve) <= max_points:
        return curve

    boundaries = np.linspace(0, len(curve), max_points + 1, dtype=np.int64)
    starts = boundaries[:-1]
    counts = np.diff(boundaries)
    time = np.add.reduceat(curve.time, starts) / counts
    flux = np.add.reduceat(curve.flux, starts) / counts
    flux_err = np.sqrt(np.add.reduceat(curve.flux_err**2, starts)) / counts
    return LightCurve(
        time=time.astype(np.float64),
        flux=flux.astype(np.float64),
        flux_err=flux_err.astype(np.float64),
        tic_id=curve.tic_id,
        sector=curve.sector,
        source=curve.source,
        metadata={
            **curve.metadata,
            "downsampled_from": len(curve),
            "downsampled_to": len(time),
        },
    )
