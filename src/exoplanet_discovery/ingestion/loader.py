"""Load validated light curves from CSV and FITS files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.io import fits

from ..common.exceptions import IngestionError, InvalidLightCurveError
from ..common.logging import get_logger
from ..common.models import LightCurve

logger = get_logger(__name__)
MAX_NAN_FRACTION = 0.2


def load_light_curve(
    source: str | Path | bytes | BinaryIO,
    filename: str | None = None,
    time_column: str | None = None,
    flux_column: str | None = None,
    flux_err_column: str | None = None,
    max_nan_fraction: float = MAX_NAN_FRACTION,
) -> LightCurve:
    """Load a CSV or FITS light curve, dropping invalid rows within quality limits."""
    name = filename or (Path(source).name if isinstance(source, (str, Path)) else "upload")
    suffix = Path(name).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(source if not isinstance(source, bytes) else BytesIO(source))
        elif suffix in {".fits", ".fit", ".fts"}:
            frame = _read_fits(source)
        else:
            raise IngestionError("Unsupported file format. Upload a CSV or FITS light curve.")
        return _frame_to_light_curve(
            frame, name, time_column, flux_column, flux_err_column, max_nan_fraction
        )
    except (IngestionError, InvalidLightCurveError):
        raise
    except Exception as exc:
        logger.warning("Failed to parse light curve upload (%s)", type(exc).__name__)
        raise IngestionError(
            "Could not read the light-curve file. Check its format and columns."
        ) from exc


def _read_fits(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = BytesIO(source)
    with fits.open(source, memmap=False) as hdus:
        for hdu in hdus:
            if hdu.data is not None and getattr(hdu.data, "names", None):
                return pd.DataFrame({name: hdu.data[name] for name in hdu.data.names})
    raise IngestionError("No table extension with light-curve columns was found in the FITS file.")


def _frame_to_light_curve(
    frame: pd.DataFrame,
    filename: str,
    time_column: str | None,
    flux_column: str | None,
    flux_err_column: str | None,
    max_nan_fraction: float,
) -> LightCurve:
    if not 0 <= max_nan_fraction < 1:
        raise ValueError("max_nan_fraction must be in [0, 1)")
    columns = {str(column).strip().lower(): str(column) for column in frame.columns}
    time_name = _find_column(columns, time_column, ("time", "bjd", "btjd", "jd"))
    flux_name = _find_column(
        columns, flux_column, ("flux", "pdcsap_flux", "sap_flux", "normalized_flux")
    )
    error_name = _find_column(
        columns, flux_err_column, ("flux_err", "pdcsap_flux_err", "sap_flux_err", "flux_error")
    )
    if time_name is None or flux_name is None:
        raise IngestionError("Required time and flux columns were not found in the file.")

    time = pd.to_numeric(frame[time_name], errors="coerce").to_numpy(dtype=float)
    flux = pd.to_numeric(frame[flux_name], errors="coerce").to_numpy(dtype=float)
    errors = (
        pd.to_numeric(frame[error_name], errors="coerce").to_numpy(dtype=float)
        if error_name is not None
        else np.full(len(frame), np.nan)
    )
    valid = np.isfinite(time) & np.isfinite(flux)
    invalid_fraction = 1 - float(np.count_nonzero(valid)) / max(len(frame), 1)
    if invalid_fraction > max_nan_fraction:
        raise InvalidLightCurveError(
            f"Too many missing or invalid time/flux values ({invalid_fraction:.1%}; "
            f"limit {max_nan_fraction:.1%})."
        )
    time, flux, errors = time[valid], flux[valid], errors[valid]
    error_valid = np.isfinite(errors) & (errors > 0)
    if not np.any(error_valid):
        errors = np.full(len(time), max(float(np.nanstd(flux)) * 0.1, 1e-8))
    else:
        fallback = float(np.median(errors[error_valid]))
        errors = np.where(error_valid, errors, fallback)
    order = np.argsort(time)
    try:
        curve = LightCurve(
            time=time[order].astype(np.float64),
            flux=flux[order].astype(np.float64),
            flux_err=errors[order].astype(np.float64),
            tic_id=Path(filename).stem[:128] or "uploaded-light-curve",
            source="uploaded",
            metadata={
                "filename": Path(filename).name,
                "time_column": time_name,
                "flux_column": flux_name,
                "source_rows": len(frame),
                "dropped_invalid_rows": int(len(frame) - np.count_nonzero(valid)),
            },
        )
    except ValueError as exc:
        raise InvalidLightCurveError(str(exc)) from exc
    logger.info("Loaded %s with %d valid cadences", Path(filename).name, len(curve))
    return curve


def _find_column(
    columns: dict[str, str], requested: str | None, candidates: tuple[str, ...]
) -> str | None:
    if requested is not None:
        return columns.get(requested.strip().lower())
    return next((columns[name] for name in candidates if name in columns), None)


def from_lightkurve(light_curve: lk.LightCurve, target_id: str = "lightkurve-target") -> LightCurve:
    """Adapt an already-loaded Lightkurve object without searching or downloading data."""
    time = np.asarray(light_curve.time.value, dtype=np.float64)
    flux = np.asarray(light_curve.flux.value, dtype=np.float64)
    raw_errors = light_curve.flux_err
    if raw_errors is None:
        errors = np.full(len(flux), max(float(np.nanstd(flux)) * 0.1, 1e-8))
    else:
        errors = np.asarray(raw_errors.value, dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(errors) & (errors > 0)
    if np.count_nonzero(valid) < 10:
        raise InvalidLightCurveError(
            "Lightkurve object has fewer than 10 valid time/flux/error rows."
        )
    order = np.argsort(time[valid])
    try:
        return LightCurve(
            time=time[valid][order],
            flux=flux[valid][order],
            flux_err=errors[valid][order],
            tic_id=str(target_id)[:128],
            source="lightkurve",
            metadata={
                "adapter": "lightkurve",
                "dropped_invalid_rows": int(len(time) - np.count_nonzero(valid)),
            },
        )
    except ValueError as exc:
        raise InvalidLightCurveError(str(exc)) from exc