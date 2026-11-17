"""GPU-independent contract tests for the API."""
import io
import time
import pytest
from PIL import Image
from fastapi.testclient import TestClient


def png():
    data = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(data, "PNG")
    return data.getvalue()


def files():
    return {"person_image": ("person.png", png(), "image/png"),
            "garment_image": ("garment.png", png(), "image/png")}


@pytest.fixture()
def client(monkeypatch):
    from app.main import app
    from app.core import config
    from app.services import inference
    monkeypatch.setattr(config, "API_TOKEN", "test-secret")
    monkeypatch.setattr(inference, "_run_inference", lambda *args: "mock-image")
    with TestClient(app) as c:
        yield c


HEADERS = {"Authorization": "Bearer test-secret"}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "gpu_available" in response.json()


def test_missing_auth(client):
    assert client.post("/api/v1/jobs", files=files()).status_code == 401


def test_invalid_auth(client):
    assert client.post("/api/v1/jobs", files=files(),
                       headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_unconfigured_auth_fails_closed(client, monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "API_TOKEN", "")
    assert client.post("/api/v1/jobs", files=files(), headers=HEADERS).status_code == 503


@pytest.mark.parametrize("field,value", [
    ("category", "not-a-category"),
    ("num_timesteps", "60"),
    ("guidance_scale", "999"),
    ("seed", "-1"),
])
def test_invalid_parameters_return_422(client, field, value):
    r = client.post("/api/v1/virtual-tryon", files=files(),
                    data={field: value}, headers=HEADERS)
    assert r.status_code == 422


def test_reject_file_type(client):
    bad = {"person_image": ("bad.txt", b"data", "text/plain"),
           "garment_image": ("good.png", png(), "image/png")}
    assert client.post("/api/v1/jobs", files=bad, headers=HEADERS).status_code == 415


def test_reject_corrupt_image(client):
    bad = {"person_image": ("bad.png", b"invalid", "image/png"),
           "garment_image": ("good.png", png(), "image/png")}
    assert client.post("/api/v1/jobs", files=bad, headers=HEADERS).status_code == 422


def test_reject_oversize_image(client, monkeypatch):
    from app.api import routes
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 20)
    assert client.post("/api/v1/jobs", files=files(), headers=HEADERS).status_code == 413


def test_no_garment(client):
    only_person = {"person_image": ("person.png", png(), "image/png")}
    assert client.post("/api/v1/jobs", files=only_person, headers=HEADERS).status_code == 422


def test_sync_endpoint(client):
    r = client.post("/api/v1/virtual-tryon", files=files(), headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["generated_image"] == "mock-image"


def test_two_garments(client):
    garment_files = {
        "person_image": ("person.png", png(), "image/png"),
        "upper_garment_image": ("top.png", png(), "image/png"),
        "lower_garment_image": ("bottom.png", png(), "image/png"),
    }
    r = client.post("/api/v1/virtual-tryon-separate", files=garment_files, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["processing_steps"] == ["upper_garment", "lower_garment"]


def test_job_submission_polling(client):
    r = client.post("/api/v1/jobs", files=files(), headers=HEADERS)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(100):
        status = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS)
        if status.json()["status"] == "completed":
            break
        time.sleep(.01)
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["generated_image"] == "mock-image"


def test_missing_job(client):
    assert client.get("/api/v1/jobs/unknown", headers=HEADERS).status_code == 404
