"""NASA mission-target ingestion with graceful offline fallback."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.table import Row
from astroquery.mast import Conf as MastConf
from astroquery.mast import Mast, Observations

from ..common.exceptions import IngestionError
from ..common.models import LightCurve
from .loader import from_lightkurve

Mission = Literal["Kepler", "TESS"]
_IS_RENDER = os.getenv("RENDER", "").lower() == "true"


class _RequestSession(Protocol):
    request: Callable[..., Any]


def _mast_timeout_seconds() -> int:
    default = "90" if _IS_RENDER else "600"
    value = os.getenv("EXOPLANET_MAST_TIMEOUT_SECONDS", default)
    try:
        timeout = int(value)
    except ValueError as exc:
        raise ValueError("EXOPLANET_MAST_TIMEOUT_SECONDS must be an integer.") from exc
    if not 1 <= timeout <= 600:
        raise ValueError("EXOPLANET_MAST_TIMEOUT_SECONDS must be between 1 and 600.")
    return timeout


MAST_REQUEST_TIMEOUT_SECONDS = _mast_timeout_seconds()


def _set_default_request_timeout(session: _RequestSession, timeout_seconds: int) -> None:
    original_request = session.request

    def request_with_timeout(method: str, url: str, **kwargs: Any) -> Any:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (10, timeout_seconds)
        return original_request(method, url, **kwargs)

    session.request = request_with_timeout


def _configure_render_mast_timeouts() -> None:
    MastConf.timeout = MAST_REQUEST_TIMEOUT_SECONDS
    sessions = (Mast._session, Observations._session)
    for index, session in enumerate(sessions):
        if any(session is earlier for earlier in sessions[:index]):
            continue
        _set_default_request_timeout(session, MAST_REQUEST_TIMEOUT_SECONDS)

    for client in (Mast, Observations):
        for connection_name in ("_portal_api_connection", "_service_api_connection"):
            connection = getattr(client, connection_name, None)
            if connection is not None:
                connection.TIMEOUT = MAST_REQUEST_TIMEOUT_SECONDS


if _IS_RENDER:
    _configure_render_mast_timeouts()


def archive_cache_dir() -> Path | None:
    configured = os.getenv("EXOPLANET_ARCHIVE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    if _IS_RENDER:
        return Path(tempfile.gettempdir()) / "exoplanet-discovery-lightkurve-cache"
    return None


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
        cache_path = Path(cache_dir).expanduser() if cache_dir is not None else None
        if cache_path is not None:
            cache_path.mkdir(parents=True, exist_ok=True)
        cache = str(cache_path) if cache_path is not None else None
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
        detail = str(exc).strip() or type(exc).__name__
        raise IngestionError(
            f"Could not access the {mission} archive. "
            f"Network/mission availability may be unavailable. Details: {detail}"
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
