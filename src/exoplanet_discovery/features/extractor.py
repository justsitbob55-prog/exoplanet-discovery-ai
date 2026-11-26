"""Astronomical feature extraction for preprocessed light curves."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle

from ..common.exceptions import FeatureExtractionError
from ..common.logging import get_logger
from ..common.models import LightCurve
from ..preprocessing.sampling import downsample_light_curve

logger = get_logger(__name__)
MAX_PERIODOGRAM_CADENCES = 20_000


def extract_features(curve: LightCurve) -> dict[str, float]:
    """Extract compact statistical, variability, and periodicity features."""
    time = np.asarray(curve.time, dtype=np.float64)
    flux = np.asarray(curve.flux, dtype=np.float64)
    err = np.asarray(curve.flux_err, dtype=np.float64)
    if len(flux) < 10:
        raise FeatureExtractionError("At least 10 cadences are required for feature extraction.")

    center = float(np.median(flux))
    mad = float(np.median(np.abs(flux - center)))
    deviations = (flux - center) / (1.4826 * mad + 1e-12)
    mean = float(np.mean(flux))
    std = float(np.std(flux))
    low = float(np.percentile(flux, 10))
    high = float(np.percentile(flux, 90))

    features = {
        "mean": mean,
        "std": std,
        "median": center,
        "mad": mad,
        "skewness": float(np.mean(deviations**3)),
        "kurtosis": float(np.mean(deviations**4) - 3.0),
        "minimum": float(np.min(flux)),
        "maximum": float(np.max(flux)),
        "range": float(np.ptp(flux)),
        "q10": low,
        "q90": high,
        "iqr": float(np.percentile(flux, 75) - np.percentile(flux, 25)),
        "flux_snr": float(std / (np.median(err) + 1e-12)),
        "cadence_count": float(len(flux)),
        "duration_days": float(time[-1] - time[0]),
        "variability_rms": float(np.sqrt(np.mean((flux - mean) ** 2))),
        "negative_outlier_fraction": float(np.mean(flux < center - 5 * mad)),
        "positive_outlier_fraction": float(np.mean(flux > center + 5 * mad)),
    }

    try:
        duration = max(float(time[-1] - time[0]), 1e-6)
        baseline = 1.0 / duration
        if len(time) > MAX_PERIODOGRAM_CADENCES:
            sampled = downsample_light_curve(curve, MAX_PERIODOGRAM_CADENCES)
            periodogram_time = sampled.time
            periodogram_flux = sampled.flux
            periodogram_err = sampled.flux_err
        else:
            periodogram_time, periodogram_flux, periodogram_err = time, flux, err
        frequency = np.linspace(
            baseline,
            min(10.0, 0.5 * len(periodogram_time) / duration),
            128,
        )
        power = LombScargle(
            periodogram_time,
            periodogram_flux,
            periodogram_err,
        ).power(frequency)
        peak = int(np.argmax(power))
        period = 1.0 / float(frequency[peak])
        features.update({
            "period_days": period,
            "periodogram_power": float(power[peak]),
            "periodogram_mean": float(np.mean(power)),
            "lowest_flux_fraction": float(np.mean(flux <= low)),
        })
        centered = flux - mean
        autocorr_zero = float(np.dot(centered, centered))
        features["autocorrelation_lag1"] = (
            float(np.dot(centered[:-1], centered[1:]) / (autocorr_zero + 1e-12))
            if len(centered) > 1
            else 0.0
        )

        differences = np.diff(flux)
        features["acf_variability"] = float(np.std(differences))
        frequencies = np.fft.rfftfreq(len(flux), d=max(duration / len(flux), 1e-6))
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        features["frequency_energy"] = float(np.sum(spectrum))
        features["dominant_frequency"] = (
            float(frequencies[int(np.argmax(spectrum[1:])) + 1])
            if len(spectrum) > 1
            else 0.0
        )
    except Exception as exc:
        logger.exception("Could not derive periodicity features for %s", curve.tic_id)
        raise FeatureExtractionError("Could not derive periodicity features.") from exc
    return {key: float(value) for key, value in features.items()}


def feature_names() -> tuple[str, ...]:
    """Return the stable feature order used for model training."""
    return (
        "mean", "std", "median", "mad", "skewness", "kurtosis", "minimum", "maximum",
        "range", "q10", "q90", "iqr", "flux_snr", "cadence_count", "duration_days",
        "variability_rms", "negative_outlier_fraction", "positive_outlier_fraction",
        "period_days", "periodogram_power", "periodogram_mean", "lowest_flux_fraction",
        "autocorrelation_lag1", "acf_variability", "frequency_energy", "dominant_frequency",
    )
