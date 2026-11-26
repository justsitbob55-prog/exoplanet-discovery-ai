"""Astronomical data ingestion components."""

from .archive import (
    create_demo_light_curve,
    download_light_curve,
    search_light_curve,
    search_targets,
    write_demo_csv,
)
from .loader import from_lightkurve, load_light_curve

__all__ = [
    "create_demo_light_curve",
    "download_light_curve",
    "from_lightkurve",
    "load_light_curve",
    "search_light_curve",
    "search_targets",
    "write_demo_csv",
]
