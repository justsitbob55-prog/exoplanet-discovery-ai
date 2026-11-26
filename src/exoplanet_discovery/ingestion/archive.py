"""NASA mission-target ingestion with graceful offline fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.table import Row

from ..common.exceptions import IngestionError
from ..common.models import LightCurve
from .loader import from_lightkurve

Mission = Literal["Kepler", "TESS"]


def _search_value(row: Row, column: str, default: str) -> str:
    try:
        value = row[column]
    except KeyError:
        return default
    if value is None or np.ma.is_masked(value):
        return default
    return str(value)


def create_demo_light_curve(target_id: str = "DEMO-TRANSIT-001") -> LightCurve:
    """Create a clearly labelled synthetic demo light curve for UI demonstrations."""
    time = np.arange(0.0, 12.0, 0.05, dtype=np.float64)
    phase = np.mod(time, 2.8)
    transit = ((phase > 1.33) & (phase < 1.46)).astype(np.float64) * 0.009
    ripple = np.sin(time * 0.31) * 0.0008 + np.cos(time * 0.73) * 0.0004
    flux = (1.0 - transit + ripple).astype(np.float64)
    flux_err = np.full(len(time), 0.0005, dtype=np.float64)
    return LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        tic_id=target_id,
        source="DEMO DATA",
        metadata={"demo": True, "filename": "synthetic_demo.csv"},
    )


def write_demo_csv(path: str | Path) -> Path:
    """Write the offline demo dataset as a reusable CSV file."""
    curve = create_demo_light_curve()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "time": curve.time,
        "flux": curve.flux,
        "flux_err": curve.flux_err,
    }).to_csv(target, index=False)
    return target


def search_light_curve(
    target: str,
    mission: Mission = "TESS",
    sector: int | None = None,
    cache_dir: str | Path | None = None,
) -> lk.LightCurve:
    """Search and download a mission light curve through Lightkurve."""
    try:
        cache = str(cache_dir) if cache_dir is not None else None
        if mission == "TESS":
            search = lk.search_lightcurve(target, mission="TESS", sector=sector, cadence="long")
        else:
            search = lk.search_lightcurve(target, mission="Kepler")
        if len(search) == 0:
            raise IngestionError(f"No {mission} light curve was found for target '{target}'.")
        collection = search.download_all(download_dir=cache)
        if collection is None or len(collection) == 0:
            raise IngestionError(f"No downloadable {mission} product was found for '{target}'.")
        combined = collection[0]
        for curve in collection[1:]:
            combined = combined.append(curve)
        return combined
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Could not access the {mission} archive. "
            "Network/mission availability may be unavailable."
        ) from exc


def download_light_curve(
    target: str,
    mission: Mission = "TESS",
    sector: int | None = None,
    cache_dir: str | Path | None = None,
) -> LightCurve:
    """Download and adapt a NASA mission light curve."""
    light_curve = search_light_curve(target, mission, sector, cache_dir)
    return from_lightkurve(light_curve, target_id=target)


def search_targets(
    target: str,
    mission: Mission = "TESS",
) -> list[dict[str, object]]:
    """Search mission products and return serializable archive metadata."""
    try:
        search = lk.search_lightcurve(target, mission=mission)
        records: list[dict[str, object]] = []
        for index, row in enumerate(search.table):
            if index >= 25:
                break
            mission_name = _search_value(row, "mission", mission)
            sector = _search_value(row, "sequence_number", "") or _search_value(
                row, "quarter", ""
            )
            if not sector and mission == "Kepler" and "Quarter " in mission_name:
                sector = mission_name.rsplit("Quarter ", maxsplit=1)[-1]
            product = _search_value(row, "productFilename", "")
            records.append({
                "index": index,
                "mission": mission_name,
                "sector": sector,
                "author": _search_value(row, "author", ""),
                "product": product or _search_value(row, "product", ""),
                "target_name": _search_value(row, "target_name", target),
            })
        return records
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise IngestionError(
            f"Archive search for '{target}' on mission '{mission}' failed: {detail}"
        ) from exc
