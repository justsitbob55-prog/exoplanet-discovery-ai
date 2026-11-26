"""FastAPI endpoints for light-curve upload and preprocessing."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import ProjectConfig
from src.exoplanet_discovery.common.exceptions import (
    IngestionError,
    InvalidLightCurveError,
    PreprocessingError,
)
from src.exoplanet_discovery.common.logging import get_logger
from src.exoplanet_discovery.common.models import LightCurve
from src.exoplanet_discovery.common.persistence import ResultStore
from src.exoplanet_discovery.detection.transit import detect_transits
from src.exoplanet_discovery.features.extractor import extract_features
from src.exoplanet_discovery.ingestion.archive import (
    create_demo_light_curve,
    download_light_curve,
    search_targets,
)
from src.exoplanet_discovery.ingestion.loader import load_light_curve
from src.exoplanet_discovery.preprocessing.pipeline import preprocess_light_curve

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
logger = get_logger(__name__)
config = ProjectConfig.default()
light_curves: dict[str, LightCurve] = {}
demo_candidates: list[dict[str, Any]] = []
store = ResultStore()
app = FastAPI(title="Exoplanet Discovery API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PreprocessRequest(BaseModel):
    light_curve_id: str


class DetectRequest(BaseModel):
    light_curve_id: str
    min_snr: float = 3.0
    confidence_threshold: float = 0.7


class TargetSearchRequest(BaseModel):
    target: str
    mission: str = "TESS"
    sector: int | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "exoplanet-discovery-api"}


@app.post("/api/light-curves/upload")
async def upload_light_curve(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    filename = file.filename or "upload"
    if not filename.lower().endswith((".csv", ".fits", ".fit", ".fts")):
        raise HTTPException(status_code=415, detail="Upload a CSV or FITS light-curve file.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB upload limit.")
    try:
        curve = load_light_curve(
            content,
            filename=filename,
            max_nan_fraction=config.data.max_nan_fraction,
        )
    except (IngestionError, InvalidLightCurveError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    curve_id = str(uuid.uuid4())
    light_curves[curve_id] = curve
    logger.info("Accepted uploaded light curve (%d cadences)", len(curve))
    return {"id": curve_id, **_serialize_curve(curve)}


@app.post("/api/light-curves/preprocess")
def preprocess(request: PreprocessRequest) -> dict[str, Any]:
    curve = light_curves.get(request.light_curve_id)
    if curve is None:
        raise HTTPException(
            status_code=404,
            detail="Light curve not found. Upload it again to continue.",
        )
    try:
        processed = preprocess_light_curve(curve, config.preprocessing)
    except PreprocessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    light_curves[request.light_curve_id] = processed
    return {"id": request.light_curve_id, **_serialize_curve(processed)}


@app.get("/api/demo")
def load_demo() -> dict[str, Any]:
    curve = create_demo_light_curve()
    curve_id = str(uuid.uuid4())
    light_curves[curve_id] = curve
    return {"id": curve_id, **_serialize_curve(curve)}


@app.post("/api/light-curves/features")
def extract_curve_features(request: PreprocessRequest) -> dict[str, Any]:
    curve = light_curves.get(request.light_curve_id)
    if curve is None:
        raise HTTPException(status_code=404, detail="Light curve not found.")
    try:
        return {"light_curve_id": request.light_curve_id, "features": extract_features(curve)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/light-curves/detect")
def detect(request: DetectRequest) -> dict[str, Any]:
    curve = light_curves.get(request.light_curve_id)
    if curve is None:
        raise HTTPException(status_code=404, detail="Light curve not found.")
    try:
        candidates = detect_transits(
            curve,
            min_snr=request.min_snr,
            confidence_threshold=request.confidence_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    serialized = [_serialize_candidate(candidate) for candidate in candidates]
    store.add_analysis(
        target_id=curve.tic_id,
        source=curve.source,
        status="AI detection completed",
        cadence_count=len(curve),
        metadata={"candidate_count": len(serialized), "official_confirmations": 0},
    )
    if serialized:
        store.add_candidates(serialized)
    global demo_candidates
    demo_candidates = serialized
    return {
        "light_curve_id": request.light_curve_id,
        "candidate_count": len(serialized),
        "candidates": serialized,
        "is_confirmed_exoplanet": False,
        "status": "AI predictions only; no official confirmation",
    }


@app.get("/api/candidates")
def get_candidates() -> dict[str, Any]:
    persisted = store.candidates()
    return {"candidates": persisted or demo_candidates, "confirmed_exoplanets": 0}


@app.post("/api/archive/search")
def archive_search(request: TargetSearchRequest) -> dict[str, Any]:
    try:
        return {
            "mission": request.mission,
            "target": request.target,
            "results": search_targets(request.target, request.mission),  # type: ignore[arg-type]
        }
    except IngestionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/archive/download")
def archive_download(request: TargetSearchRequest) -> dict[str, Any]:
    try:
        curve = download_light_curve(
            request.target,
            request.mission,  # type: ignore[arg-type]
            sector=request.sector,
            cache_dir=config.data.external_data_dir,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    curve_id = str(uuid.uuid4())
    light_curves[curve_id] = curve
    store.add_analysis(
        target_id=curve.tic_id,
        source=f"{request.mission} archive",
        status="downloaded",
        cadence_count=len(curve),
        metadata={"sector": request.sector},
    )
    return {"id": curve_id, **_serialize_curve(curve)}


@app.get("/api/history")
def analysis_history() -> dict[str, Any]:
    return {"analyses": store.analyses()}


@app.get("/api/candidates/export")
def export_candidates() -> FileResponse:
    target = Path("data/exports/candidate_results.csv")
    store.candidate_csv(target)
    return FileResponse(
        target,
        media_type="text/csv",
        filename="exoplanet_candidate_results.csv",
    )


@app.get("/api/light-curves/{light_curve_id}")
def get_light_curve(light_curve_id: str) -> dict[str, Any]:
    curve = light_curves.get(light_curve_id)
    if curve is None:
        raise HTTPException(status_code=404, detail="Light curve not found.")
    return {"id": light_curve_id, **_serialize_curve(curve)}


def _serialize_curve(curve: LightCurve) -> dict[str, Any]:
    return {
        "name": str(curve.metadata.get("filename", curve.tic_id)),
        "source": curve.source,
        "cadence_count": len(curve),
        "duration_days": curve.duration_days,
        "time": curve.time.tolist(),
        "flux": curve.flux.tolist(),
        "flux_err": curve.flux_err.tolist(),
        "metadata": curve.metadata,
    }


def _serialize_candidate(candidate: Any) -> dict[str, Any]:
    transit = candidate.best_transit
    return {
        "target_id": candidate.tic_id,
        "candidate_name": candidate.candidate_name,
        "classification": "AI_PREDICTED_CANDIDATE",
        "confirmed_exoplanet": False,
        "probability": candidate.exoplanet_probability,
        "period_days": candidate.estimated_period_days,
        "transit_depth_ppm": transit.transit_depth_ppm if transit else None,
        "transit_duration_hours": transit.duration_hours if transit else None,
        "snr": transit.snr if transit else None,
        "confidence": transit.confidence if transit else candidate.exoplanet_probability,
        "notes": candidate.notes,
    }