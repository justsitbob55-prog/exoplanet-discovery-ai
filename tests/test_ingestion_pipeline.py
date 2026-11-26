"""Focused tests for light-curve ingestion and preprocessing."""

from io import BytesIO

import lightkurve as lk
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import MaskedColumn, Table

from src.exoplanet_discovery.common.exceptions import IngestionError, InvalidLightCurveError
from src.exoplanet_discovery.common.models import LightCurve
from src.exoplanet_discovery.detection.transit import detect_transits
from src.exoplanet_discovery.ingestion import archive
from src.exoplanet_discovery.ingestion.archive import create_demo_light_curve
from src.exoplanet_discovery.ingestion.loader import from_lightkurve, load_light_curve
from src.exoplanet_discovery.preprocessing.pipeline import preprocess_light_curve


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def test_load_csv_sorts_rows_and_fills_uncertainties() -> None:
    frame = pd.DataFrame({"time": np.arange(20)[::-1], "flux": np.ones(20)})
    curve = load_light_curve(_csv_bytes(frame), filename="sample.csv")

    assert len(curve) == 20
    assert np.all(np.diff(curve.time) > 0)
    assert np.all(curve.flux_err > 0)


def test_load_csv_drops_limited_missing_rows() -> None:
    frame = pd.DataFrame({"time": np.arange(20), "flux": np.ones(20)})
    frame.loc[0, "flux"] = np.nan

    assert len(load_light_curve(_csv_bytes(frame), filename="sample.csv")) == 19


def test_load_csv_rejects_excessive_missing_rows() -> None:
    frame = pd.DataFrame({"time": np.arange(20), "flux": [np.nan] * 5 + [1.0] * 15})

    with pytest.raises(InvalidLightCurveError, match="Too many"):
        load_light_curve(_csv_bytes(frame), filename="sample.csv")


def test_unsupported_format_has_actionable_error() -> None:
    with pytest.raises(IngestionError, match="CSV or FITS"):
        load_light_curve(b"data", filename="sample.txt")


def test_load_fits_table() -> None:
    buffer = BytesIO()
    columns = [
        fits.Column(name="TIME", format="D", array=np.arange(20, dtype=float)),
        fits.Column(name="PDCSAP_FLUX", format="D", array=np.ones(20)),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(buffer)

    assert len(load_light_curve(buffer.getvalue(), filename="sample.fits")) == 20


def test_adapt_loaded_lightkurve_object() -> None:
    source = lk.LightCurve(
        time=np.arange(20, dtype=float)[::-1],
        flux=np.ones(20),
        flux_err=np.full(20, 0.01),
    )

    curve = from_lightkurve(source, target_id="TIC-123")

    assert curve.tic_id == "TIC-123"
    assert curve.source == "lightkurve"
    assert np.all(np.diff(curve.time) > 0)


def test_search_targets_reads_lightkurve_table_and_passes_mission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = Table({
        "mission": ["Kepler Quarter 01"],
        "sequence_number": MaskedColumn([0], mask=[True]),
        "author": ["Kepler"],
        "productFilename": ["kplr011442793-2009166043257_llc.fits"],
        "target_name": ["kplr011442793"],
    })
    calls: list[tuple[str, str]] = []

    def fake_search_lightcurve(target: str, *, mission: str) -> object:
        calls.append((target, mission))
        return type("SearchResultStub", (), {"table": table})()

    monkeypatch.setattr(archive.lk, "search_lightcurve", fake_search_lightcurve)

    results = archive.search_targets("Kepler-90", "Kepler")

    assert calls == [("Kepler-90", "Kepler")]
    assert results == [{
        "index": 0,
        "mission": "Kepler Quarter 01",
        "sector": "01",
        "author": "Kepler",
        "product": "kplr011442793-2009166043257_llc.fits",
        "target_name": "kplr011442793",
    }]


def test_gradio_archive_search_reports_no_products_and_passes_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import gradio_app

    calls: list[tuple[str, str]] = []

    def fake_search_targets(target: str, mission: str) -> list[dict[str, object]]:
        calls.append((target, mission))
        return []

    monkeypatch.setattr(gradio_app, "search_targets", fake_search_targets)

    result = gradio_app.archive_search(" Kepler-90 ", "Kepler")

    assert calls == [("Kepler-90", "Kepler")]
    assert result.to_dict(orient="records") == [{
        "message": "No Kepler light-curve products found for 'Kepler-90'."
    }]


def test_search_targets_preserves_archive_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_search_lightcurve(target: str, *, mission: str) -> object:
        raise RuntimeError("MAST request timed out")

    monkeypatch.setattr(archive.lk, "search_lightcurve", fail_search_lightcurve)

    with pytest.raises(IngestionError) as error:
        archive.search_targets("Kepler-90", "Kepler")

    assert "Kepler-90" in str(error.value)
    assert "Kepler" in str(error.value)
    assert "MAST request timed out" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_preprocessing_removes_spike_detrends_and_normalizes() -> None:
    time = np.linspace(0, 10, 100)
    flux = 1000 + time * 0.5
    flux[50] = 1400
    curve = load_light_curve(
        _csv_bytes(pd.DataFrame({"time": time, "flux": flux})), filename="trend.csv"
    )

    processed = preprocess_light_curve(curve)

    assert len(processed) == 99
    assert np.median(processed.flux) == pytest.approx(1.0, abs=0.02)
    assert processed.metadata["removed_outliers"] == 1


def test_synthetic_demo_runs_full_transit_detection() -> None:
    raw = create_demo_light_curve()
    processed = preprocess_light_curve(raw)

    candidates = detect_transits(
        processed,
        min_snr=0.5,
        confidence_threshold=0.1,
    )

    assert len(candidates) == 1
    transit = candidates[0].best_transit
    assert transit is not None
    assert transit.transit_depth_ppm > 0
    assert transit.snr >= 0
    assert candidates[0].exoplanet_probability >= 0.1


def test_gradio_download_returns_preview_status_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import gradio_app

    curve = create_demo_light_curve("TIC-DOWNLOADED")

    def fake_download(target: str, mission: str, sector: int | None = None) -> object:
        assert target == "TIC-123"
        assert mission == "TESS"
        assert sector == 7
        return curve

    monkeypatch.setattr(gradio_app, "download_light_curve", fake_download)

    frame, status, state = gradio_app.archive_download(" TIC-123 ", "TESS", "7")

    assert len(frame) == len(curve)
    assert "TIC-123" in status
    assert state is curve


def test_gradio_download_caps_large_preview_but_keeps_full_curve_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import gradio_app

    time = np.linspace(0.0, 27.0, 12_000)
    curve = LightCurve(
        time,
        np.ones_like(time),
        np.full_like(time, 0.001),
        "TIC-LARGE-PREVIEW",
    )
    monkeypatch.setattr(gradio_app, "download_light_curve", lambda *args, **kwargs: curve)

    preview, status, state = gradio_app.archive_download("TIC-123", "TESS", "")

    assert len(preview) == gradio_app.MAX_PLOT_CADENCES
    assert "12000 cadences" in status
    assert state is curve


def test_gradio_analysis_prefers_downloaded_curve_over_demo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import gradio_app

    class StoreStub:
        def add_analysis(self, *args: object, **kwargs: object) -> None:
            return None

        def add_candidates(self, *args: object, **kwargs: object) -> None:
            return None

    downloaded = create_demo_light_curve("TIC-DOWNLOADED")
    monkeypatch.setattr(gradio_app, "store", StoreStub())
    monkeypatch.setattr(
        gradio_app,
        "detect_transits",
        lambda curve, **kwargs: [],
    )

    _, _, _, candidates, source, status = gradio_app.run_analysis_with_state(
        None,
        True,
        downloaded,
        "TIC-DOWNLOADED",
        "TESS",
    )

    assert "TESS archive download" in source
    assert "no transit candidates" in status.lower()
    assert candidates.to_dict(orient="records")[0]["message"].startswith("No candidate")