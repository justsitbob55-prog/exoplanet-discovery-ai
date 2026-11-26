"""Quality filtering and normalization for light curves."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from src.config import PreprocessingConfig

from ..common.exceptions import PreprocessingError
from ..common.logging import get_logger
from ..common.models import LightCurve

logger = get_logger(__name__)


def preprocess_light_curve(
    curve: LightCurve, config: PreprocessingConfig | None = None
) -> LightCurve:
    """Filter robust flux outliers, remove slow trends, and normalize flux."""
    settings = config or PreprocessingConfig()
    time = np.asarray(curve.time, dtype=np.float64)
    flux = np.asarray(curve.flux, dtype=np.float64)
    flux_err = np.asarray(curve.flux_err, dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]
    if len(time) < 10:
        raise PreprocessingError("Fewer than 10 finite cadences are available.")

    if np.any(np.diff(time) < 0):
        order = np.argsort(time, kind="stable")
        time, flux, flux_err = time[order], flux[order], flux_err[order]
    duplicate_starts = np.flatnonzero(np.diff(time) != 0) + 1
    if len(duplicate_starts) < len(time) - 1:
        starts = np.concatenate(([0], duplicate_starts))
        counts = np.diff(np.append(starts, len(time)))
        removed_duplicates = len(time) - len(starts)
        flux = np.add.reduceat(flux, starts) / counts
        flux_err = np.sqrt(np.add.reduceat(flux_err**2, starts)) / counts
        time = time[starts]
    else:
        removed_duplicates = 0

    center = float(np.median(flux))
    mad = float(np.median(np.abs(flux - center)))
    scale = 1.4826 * mad
    keep = (
        np.ones(len(flux), dtype=bool)
        if scale == 0 or not settings.sigma_clip
        else flux <= center + settings.outlier_sigma * scale
    )
    if np.count_nonzero(keep) < 10:
        raise PreprocessingError("Fewer than 10 valid points remain after outlier filtering.")
    time, flux, flux_err = time[keep], flux[keep], flux_err[keep]

    if settings.detrend and len(flux) > settings.polynomial_order + 2:
        degree = min(settings.polynomial_order, len(flux) - 2)
        coordinates = np.linspace(-1.0, 1.0, len(time))
        trend = np.polyval(np.polyfit(coordinates, flux, degree), coordinates)
        baseline = float(np.median(trend))
        if not np.isfinite(baseline) or abs(baseline) < 1e-12:
            raise PreprocessingError("Detrending produced an invalid baseline.")
        flux = flux / trend * baseline

    if settings.noise_reduction_enabled and len(flux) >= 5:
        window = min(settings.detrend_window, len(flux) if len(flux) % 2 else len(flux) - 1)
        window = max(5, window if window % 2 else window - 1)
        if window <= len(flux):
            flux = savgol_filter(flux, window_length=window, polyorder=min(2, window - 1))

    if settings.normalize:
        normalizer = float(np.nanpercentile(flux, settings.normalization_percentile))
        if not np.isfinite(normalizer) or abs(normalizer) < 1e-12:
            raise PreprocessingError("Flux cannot be normalized because its baseline is zero.")
        flux = flux / normalizer
        flux_err = flux_err / abs(normalizer)

    try:
        result = LightCurve(
            time=time.astype(np.float64),
            flux=flux.astype(np.float64),
            flux_err=flux_err.astype(np.float64),
            tic_id=curve.tic_id,
            sector=curve.sector,
            source=curve.source,
            metadata={
                **curve.metadata,
                "preprocessed": True,
                "removed_outliers": int(len(keep) - np.count_nonzero(keep)),
                "removed_duplicates": int(removed_duplicates),
            },
        )
    except ValueError as exc:
        raise PreprocessingError(str(exc)) from exc
    logger.info(
        "Preprocessed %s: %d/%d cadences retained (%d invalid, %d duplicates)",
        curve.tic_id,
        len(result),
        len(curve),
        int(len(curve) - np.count_nonzero(valid)),
        removed_duplicates,
    )
    return result