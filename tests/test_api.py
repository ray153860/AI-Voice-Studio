import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from voice_studio import app as app_module


def test_project_and_audio_job(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    monkeypatch.setattr(app_module, "INBOX", tmp_path / "audio")
    monkeypatch.setattr(app_module, "OUT", tmp_path / "outputs")
    monkeypatch.setattr(app_module, "MODELS", tmp_path / "models")
    app_module.INBOX.mkdir()
    app_module.OUT.mkdir()
    app_module.MODELS.mkdir()
    from voice_studio.store import Store
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    app_module.jobs.clear()

    client = TestClient(app_module.app)
    created = client.post("/api/projects", json={"name": "Demo"})
    assert created.status_code == 200
    assert client.get("/api/projects").json()[0]["name"] == "Demo"

    buf = io.BytesIO()
    sf.write(buf, (np.sin(np.arange(8000) * .03) * .2).astype("float32"), 8000, format="WAV")
    response = client.post("/api/workflow/audio", files={"file": ("tone.wav", buf.getvalue(), "audio/wav")})
    assert response.status_code == 200
    jid = response.json()["id"]
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "completed"
    assert client.get(job["output"]).status_code == 200
    assert client.get("/api/jobs").json()[0]["id"] == jid

    retried = client.post(f"/api/jobs/{jid}/retry")
    assert retried.status_code == 200
    retried_job = client.get(f"/api/jobs/{retried.json()['id']}").json()
    assert retried_job["status"] == "completed"
    assert retried_job["log"][0] == f"retried from {jid}"
    assert client.post(f"/api/jobs/{retried_job['id']}/cancel").status_code == 409

    model = client.post(
        "/api/models",
        data={"model_type": "rvc", "license_name": "user-provided", "source_url": "https://example.test/model"},
        files={"file": ("voice.pth", b"test model bytes")},
    )
    assert model.status_code == 200
    assert model.json()["status"] == "registered_unverified"
    assert model.json()["sha256"]
    assert client.get("/api/models").json()[0]["id"] == model.json()["id"]
    assert client.get("/api/capabilities").json()["voice_conversion"] == "interface_only_dependencies_missing"


def test_invalid_audio_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    client = TestClient(app_module.app)
    assert client.post("/api/workflow/audio", files={"file": ("bad.wav", b"not audio")}).status_code == 400


def test_cancel_queued_job(tmp_path, monkeypatch):
    from voice_studio.store import Store

    store = Store(tmp_path / "studio.sqlite3")
    monkeypatch.setattr(app_module, "store", store)
    app_module.jobs.clear()
    app_module.cancel_flags.clear()
    job = {"id": "queued-job", "status": "queued", "input": "tone.wav", "metadata": {}, "log": []}
    app_module.jobs[job["id"]] = job
    store.save_job(job)

    client = TestClient(app_module.app)
    response = client.post("/api/jobs/queued-job/cancel")
    assert response.status_code == 200
    assert app_module.cancel_flags["queued-job"] is True
    assert store.get_job("queued-job")["log"] == ["cancellation requested"]


def test_model_import_validation(tmp_path, monkeypatch):
    from voice_studio.store import Store

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(app_module, "MODELS", models)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    client = TestClient(app_module.app)

    wrong_extension = client.post(
        "/api/models",
        data={"model_type": "rvc", "license_name": "unknown"},
        files={"file": ("voice.exe", b"bad")},
    )
    assert wrong_extension.status_code == 400
    missing_license = client.post(
        "/api/models",
        data={"model_type": "rvc", "license_name": " "},
        files={"file": ("voice.pth", b"data")},
    )
    assert missing_license.status_code == 400
