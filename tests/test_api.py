"""Tests for the light-curve API validation and lifecycle."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app, light_curves

client = TestClient(app)


def setup_function() -> None:
    light_curves.clear()


def test_health_endpoint() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_rejects_unsupported_format() -> None:
    response = client.post(
        "/api/light-curves/upload", files={"file": ("notes.txt", b"not a curve")}
    )

    assert response.status_code == 415
    assert "CSV or FITS" in response.json()["detail"]


def test_upload_and_preprocess_curve() -> None:
    csv = "time,flux\n" + "".join(f"{index},{1 + index * 0.0001}\n" for index in range(30))
    uploaded = client.post(
        "/api/light-curves/upload", files={"file": ("curve.csv", csv.encode())}
    )
    curve_id = uploaded.json()["id"]
    processed = client.post(
        "/api/light-curves/preprocess", json={"light_curve_id": curve_id}
    )

    assert uploaded.status_code == 200
    assert processed.status_code == 200
    assert processed.json()["metadata"]["preprocessed"] is True
    assert np.median(processed.json()["flux"]) == pytest.approx(1.0, abs=0.02)


def test_preprocess_unknown_curve_returns_404() -> None:
    response = client.post(
        "/api/light-curves/preprocess", json={"light_curve_id": "missing"}
    )

    assert response.status_code == 404