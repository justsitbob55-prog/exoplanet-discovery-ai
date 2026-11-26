"""Tests for feature, training, detection, and demo pipeline surfaces."""

from io import BytesIO
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.exoplanet_discovery.common.models import LightCurve
from src.exoplanet_discovery.common.persistence import ResultStore
from src.exoplanet_discovery.detection.transit import detect_transits
from src.exoplanet_discovery.features.extractor import extract_features, feature_names
from src.exoplanet_discovery.ingestion.archive import create_demo_light_curve
from src.exoplanet_discovery.preprocessing.pipeline import preprocess_light_curve
from src.exoplanet_discovery.preprocessing.sampling import downsample_light_curve
from src.exoplanet_discovery.training.baseline import train_baseline


def test_demo_pipeline_features_and_detection_are_labelled() -> None:
    raw = create_demo_light_curve()
    assert raw.source == "DEMO DATA"
    assert raw.metadata["demo"] is True

    processed = preprocess_light_curve(raw)
    features = extract_features(processed)
    assert set(feature_names()).issubset(features)
    assert all(np.isfinite(value) for value in features.values())


def test_downsampling_preserves_binned_transit_depth() -> None:
    time = np.linspace(0, 20, 20_000)
    flux = np.ones_like(time)
    flux[5_000:5_100] = 0.99
    curve = LightCurve(
        time=time,
        flux=flux,
        flux_err=np.full_like(time, 0.001),
        tic_id="TIC-BIN-TEST",
    )

    sampled = downsample_light_curve(curve, max_points=2_000)

    assert len(sampled) == 2_000
    np.testing.assert_allclose(np.min(sampled.flux), 0.99, atol=1e-8)


def test_large_feature_extraction_avoids_quadratic_autocorrelation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 899_056
    time = np.linspace(0.0, 27.0, count)
    flux = 1.0 + 0.001 * np.sin(2 * np.pi * time / 2.8)
    curve = LightCurve(
        time=time,
        flux=flux,
        flux_err=np.full(count, 0.0005),
        tic_id="TIC-LARGE-TEST",
    )

    def reject_quadratic_correlate(*args: object, **kwargs: object) -> None:
        raise AssertionError("Feature extraction must not call quadratic np.correlate.")

    monkeypatch.setattr(np, "correlate", reject_quadratic_correlate)
    processed = preprocess_light_curve(curve)
    features = extract_features(processed)
    candidates = detect_transits(processed, feature_values=features)

    assert all(np.isfinite(value) for value in features.values())
    assert features["cadence_count"] == count
    assert len(processed) == count
    assert isinstance(candidates, list)


def test_preprocessing_removes_duplicates_without_clipping_transit_dips() -> None:
    time = np.arange(100, dtype=np.float64) * 0.1
    flux = np.ones(100, dtype=np.float64)
    flux[40:46] = 0.98
    time = np.insert(time, 43, time[42])
    flux = np.insert(flux, 43, 0.98)
    error = np.full_like(time, 0.001)
    curve = LightCurve(time, flux, error, "TIC-DUPLICATES")

    processed = preprocess_light_curve(curve)

    assert len(processed) == 100
    assert processed.metadata["removed_duplicates"] == 1
    assert np.min(processed.flux) < 0.99


def test_gradio_reuses_cached_preprocessing_and_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import gradio_app

    raw = create_demo_light_curve("TIC-CACHE-TEST")
    calls = {"preprocess": 0, "features": 0}
    original_preprocess = gradio_app.preprocess_light_curve
    original_features = gradio_app.extract_features

    def count_preprocess(curve: LightCurve) -> LightCurve:
        calls["preprocess"] += 1
        return original_preprocess(curve)

    def count_features(curve: LightCurve) -> dict[str, float]:
        calls["features"] += 1
        return original_features(curve)

    monkeypatch.setattr(gradio_app, "preprocess_light_curve", count_preprocess)
    monkeypatch.setattr(gradio_app, "extract_features", count_features)
    progress = gradio_app.gr.Progress()

    first = gradio_app._prepare_analysis(raw, progress)
    second = gradio_app._prepare_analysis(raw, progress)

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert calls == {"preprocess": 1, "features": 1}


def test_gradio_callback_returns_all_outputs_and_closes_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import gradio_app

    progress_updates: list[float] = []

    def record_progress(
        progress: gr.Progress,
        value: float,
        *,
        desc: str | None = None,
    ) -> None:
        progress_updates.append(value)

    class StoreStub:
        def add_analysis(self, *args: object, **kwargs: object) -> None:
            return None

        def add_candidates(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(gradio_app, "store", StoreStub())
    monkeypatch.setattr(gr.Progress, "__call__", record_progress)
    monkeypatch.setattr(gradio_app, "_prepare_analysis", lambda curve, progress: (
        curve,
        {"periodogram_power": 0.5},
    ))
    monkeypatch.setattr(gradio_app, "detect_transits", lambda *args, **kwargs: [])
    open_figures = set(plt.get_fignums())

    result = gradio_app.run_analysis_with_state(
        None, True, None, "TIC-CALLBACK-TEST", "TESS", gr.Progress()
    )

    assert len(result) == 6
    assert result[4] == "DEMO DATA - synthetic demonstration"
    assert "no transit candidates" in result[5].lower()
    assert progress_updates[-1] == 1.0
    assert set(plt.get_fignums()) == open_figures
    assert isinstance(result[0], Figure)
    result[0].savefig(BytesIO(), format="png")


def test_gradio_callback_returns_error_outputs_and_finishes_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import gradio_app

    progress_updates: list[float] = []

    def record_progress(
        progress: gr.Progress,
        value: float,
        *,
        desc: str | None = None,
    ) -> None:
        progress_updates.append(value)

    def fail_load(*args: object, **kwargs: object) -> tuple[object, str]:
        raise ValueError("invalid input")

    monkeypatch.setattr(gr.Progress, "__call__", record_progress)
    monkeypatch.setattr(gradio_app, "_load", fail_load)
    open_figures = set(plt.get_fignums())

    result = gradio_app.run_analysis_with_state(
        None, True, None, "", "TESS", gr.Progress()
    )

    assert len(result) == 6
    assert result[5] == "Analysis failed: ValueError: invalid input"
    assert progress_updates[-1] == 1.0
    assert set(plt.get_fignums()) == open_figures


def test_gradio_runs_uploaded_and_demo_data_through_candidate_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import gradio_app

    class StoreStub:
        def add_analysis(self, *args: object, **kwargs: object) -> None:
            return None

        def add_candidates(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(gradio_app, "store", StoreStub())
    open_figures = set(plt.get_fignums())

    for file, use_demo, expected_source in (
        (Path("data/offline/synthetic_transit.csv"), False, "Uploaded data:"),
        (None, True, "DEMO DATA"),
    ):
        result = gradio_app.run_analysis_with_state(
            file, use_demo, None, "", "TESS", gr.Progress()
        )

        assert result[4].startswith(expected_source)
        assert "candidate(s) found" in result[5]
        assert "periodogram_power" in set(result[2]["feature"])
        assert result[3].iloc[0]["candidate_name"].startswith("AI-PREDICTED-")
        assert not bool(result[3].iloc[0]["confirmed_exoplanet"])
        assert isinstance(result[0], Figure)
        assert isinstance(result[1], Figure)
    assert set(plt.get_fignums()) == open_figures


def test_baseline_training_is_reproducible_and_returns_metrics() -> None:
    features = np.vstack([
        np.array([1.0, 0.1, 0.2]),
        np.array([1.1, 0.2, 0.1]),
        np.array([1.2, 0.1, 0.3]),
        np.array([4.0, 2.0, 1.5]),
        np.array([4.2, 2.1, 1.4]),
        np.array([4.1, 1.9, 1.6]),
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])

    first = train_baseline(features, labels, ["a", "b", "c"])
    second = train_baseline(features, labels, ["a", "b", "c"])
    assert first.metrics["accuracy"] == second.metrics["accuracy"]
    assert first.train_size == 4
    assert first.test_size == 2


def test_sqlite_store_persists_analysis_and_candidates(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "results.db")
    analysis_id = store.add_analysis(
        "TIC-DEMO",
        "DEMO DATA",
        "analysis completed",
        240,
        {"synthetic": True},
    )
    assert analysis_id > 0
    store.add_candidates([{
        "target_id": "TIC-DEMO",
        "candidate_name": "AI-PREDICTED-TIC-DEMO",
        "classification": "AI_PREDICTED_CANDIDATE",
        "confirmed_exoplanet": False,
        "probability": 0.8,
        "period_days": 2.8,
        "transit_depth_ppm": 9000.0,
        "transit_duration_hours": 3.1,
        "snr": 8.0,
        "confidence": 0.8,
        "notes": "Synthetic unverified prediction",
    }])
    assert len(store.analyses()) == 1
    assert len(store.candidates()) == 1
    export = store.candidate_csv(tmp_path / "candidates.csv")
    assert export.exists()
