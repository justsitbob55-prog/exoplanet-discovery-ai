"""Scientific transit screening and candidate construction."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares

from ..common.exceptions import FeatureExtractionError
from ..common.logging import get_logger
from ..common.models import ExoplanetCandidate, LightCurve, TransitEvent
from ..features.extractor import extract_features, feature_names
from ..preprocessing.sampling import downsample_light_curve

logger = get_logger(__name__)
MAX_BLS_CADENCES = 50_000
MAX_BLS_PERIODS = 4_096


def _trained_model_probability(features: dict[str, float]) -> float | None:
    model_path = Path("models/checkpoints/baseline_random_forest.pkl")
    metadata_path = Path("models/checkpoints/baseline_random_forest.json")
    if not model_path.exists():
        return None
    try:
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        trained_names = feature_names()
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            trained_names = tuple(metadata.get("feature_names", trained_names))
        missing = [name for name in trained_names if name not in features]
        if missing:
            raise FeatureExtractionError(
                f"Trained model requires missing features: {', '.join(missing)}"
            )
        matrix = np.asarray([[features[name] for name in trained_names]], dtype=np.float64)
        probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
        classes = getattr(model, "classes_", None)
        if classes is None:
            return float(probabilities[0, -1])
        positive = np.flatnonzero(np.asarray(classes) == 1)
        index = int(positive[0]) if len(positive) else probabilities.shape[1] - 1
        probability = float(np.clip(probabilities[0, index], 0.0, 1.0))
        logger.info("Applied trained Random Forest probability %.4f", probability)
        return probability
    except Exception as exc:
        logger.exception("Trained model loading or prediction failed")
        raise FeatureExtractionError(
            f"Trained model could not be used: {type(exc).__name__}: {exc}"
        ) from exc


def detect_transits(
    curve: LightCurve,
    min_snr: float = 3.0,
    confidence_threshold: float = 0.7,
    feature_values: dict[str, float] | None = None,
) -> list[ExoplanetCandidate]:
    """Run a Box Least Squares screen and return prediction-only candidates."""
    observation_duration = max(float(curve.time[-1] - curve.time[0]), 1e-6)
    if observation_duration < 0.5:
        raise FeatureExtractionError("The observed time span is too short for transit screening.")

    sampled = downsample_light_curve(curve, MAX_BLS_CADENCES)
    time = np.asarray(sampled.time, dtype=np.float64)
    flux = np.asarray(sampled.flux, dtype=np.float64)
    err = np.asarray(sampled.flux_err, dtype=np.float64)
    median = float(np.median(flux))
    if not np.isfinite(median) or abs(median) < 1e-12:
        raise FeatureExtractionError("Transit screening requires a non-zero flux baseline.")
    normalized = flux / median
    try:
        bls = BoxLeastSquares(time, normalized, err / median)
        minimum_period = max(0.5, 2 * float(np.median(np.diff(time))))
        maximum_period = max(minimum_period * 1.01, observation_duration / 2)
        frequency = np.linspace(
            1.0 / maximum_period,
            1.0 / minimum_period,
            MAX_BLS_PERIODS,
        )
        periods = 1.0 / frequency
        minimum_duration = max(0.02, 2 * float(np.median(np.diff(time))))
        maximum_duration = min(0.5, minimum_period * 0.8, observation_duration / 20)
        if maximum_duration < minimum_duration:
            minimum_duration = maximum_duration
        durations = np.geomspace(minimum_duration, maximum_duration, 8)
        result = bls.power(
            periods,
            durations,
            method="fast",
            oversample=5,
        )
        peak = int(np.argmax(result.power))
        period = float(result.period[peak])
        transit_duration = float(result.duration[peak])
        depth = max(float(result.depth[peak]), 0.0)
        snr = float(result.depth_snr[peak])
    except Exception as exc:
        raise FeatureExtractionError("Box Least Squares analysis failed.") from exc

    depth_ppm = max(depth * 1_000_000.0, 0.0)
    bls_confidence = (
        float(np.clip(1.0 - np.exp(-(snr - min_snr) / 3.0), 0.0, 1.0))
        if snr >= min_snr
        else 0.0
    )
    features = feature_values if feature_values is not None else extract_features(curve)
    model_probability = _trained_model_probability(features)
    if model_probability is None:
        confidence = bls_confidence
        model_note = "trained model unavailable; BLS confidence used"
    else:
        confidence = float(np.clip((bls_confidence + model_probability) / 2.0, 0.0, 1.0))
        model_note = f"trained model probability={model_probability:.4f}"
    logger.info(
        "BLS result: period=%.4f days, depth=%.1f ppm, snr=%.3f, confidence=%.4f",
        period,
        depth_ppm,
        snr,
        confidence,
    )
    if confidence < confidence_threshold:
        return []

    transit = TransitEvent(
        transit_time=float(time[np.argmin(normalized)]),
        transit_depth_ppm=depth_ppm,
        transit_depth_err=float(np.median(err) / median * 1_000_000),
        duration_hours=transit_duration * 24,
        period_days=period,
        snr=snr,
        confidence=confidence,
    )
    candidate = ExoplanetCandidate(
        tic_id=curve.tic_id,
        candidate_name=f"AI-PREDICTED-{curve.tic_id}",
        exoplanet_probability=confidence,
        transit_events=[transit],
        estimated_period_days=period,
        notes=(
            f"Unverified prediction; BLS depth={depth_ppm:.1f} ppm; "
            f"periodogram_power={features['periodogram_power']:.4f}; {model_note}"
        ),
    )
    return [candidate]


def rank_candidates(candidates: list[ExoplanetCandidate]) -> list[ExoplanetCandidate]:
    """Rank predictions without changing their predicted/confirmed status."""
    return sorted(candidates, key=lambda item: item.exoplanet_probability, reverse=True)
