import io
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from voice_studio import app as app_module
from voice_studio.engines import EngineResult


def test_home_exposes_diffsinger_submission_controls():
    response = TestClient(app_module.app).get("/")

    assert response.status_code == 200
    assert 'id="singingDs"' in response.text
    assert 'id="singingExperiment"' in response.text
    assert "提交 DiffSinger 任务" in response.text


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


def test_tts_requires_configured_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("GPT_SOVITS_URL", raising=False)
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    client = TestClient(app_module.app)
    response = client.post(
        "/api/workflow/tts",
        data={"text": "测试", "text_lang": "zh", "prompt_lang": "zh"},
        files={"reference_audio": ("reference.wav", b"not read because engine is unavailable")},
    )
    assert response.status_code == 503
    assert client.get("/api/engines").json()[0]["configured"] is False


def test_tts_workflow_with_simulated_adapter(tmp_path, monkeypatch):
    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    inbox.mkdir()
    outputs.mkdir()
    monkeypatch.setenv("GPT_SOVITS_URL", "http://engine.test:9880")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", __import__("voice_studio.store", fromlist=["Store"]).Store(tmp_path / "studio.sqlite3"))
    app_module.jobs.clear()
    app_module.cancel_flags.clear()

    class FakeAdapter:
        def __init__(self, base_url, output_dir):
            self.output_dir = output_dir

        async def run(self, request):
            target = self.output_dir / "simulated.wav"
            sf.write(target, np.zeros(2400, dtype="float32"), 24000)
            return EngineResult(str(target), 24000, "simulated-gpt-sovits", {"duration": 0.1})

    monkeypatch.setattr(app_module, "GptSovitsAdapter", FakeAdapter)
    reference = io.BytesIO()
    sf.write(reference, np.zeros(8000, dtype="float32"), 16000, format="WAV")
    client = TestClient(app_module.app)
    response = client.post(
        "/api/workflow/tts",
        data={"text": "测试", "text_lang": "zh", "prompt_lang": "zh"},
        files={"reference_audio": ("reference.wav", reference.getvalue(), "audio/wav")},
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["result"]["engine"] == "simulated-gpt-sovits"
    assert client.get(job["output"]).status_code == 200


def test_voice_conversion_requires_configured_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("APPLIO_ROOT", raising=False)
    monkeypatch.delenv("APPLIO_PYTHON", raising=False)
    client = TestClient(app_module.app)
    response = client.post(
        "/api/workflow/voice-conversion",
        data={"model_id": "missing"},
        files={"input_audio": ("input.wav", b"unused")},
    )
    assert response.status_code == 503


def test_voice_conversion_workflow_with_simulated_adapter(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    engine = tmp_path / "applio"
    for directory in (inbox, outputs, engine):
        directory.mkdir()
    model_path = tmp_path / "voice.pth"
    model_path.write_bytes(b"model")
    store = Store(tmp_path / "studio.sqlite3")
    store.save_model(
        {"id": "voice-model", "type": "rvc", "name": "voice.pth", "path": str(model_path), "status": "registered_unverified"}
    )
    monkeypatch.setenv("APPLIO_ROOT", str(engine))
    monkeypatch.setenv("APPLIO_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", store)
    app_module.jobs.clear()
    app_module.cancel_flags.clear()

    class FakeAdapter:
        def __init__(self, applio_root, python_executable, output_dir):
            self.output_dir = output_dir

        async def run(self, request):
            assert request.model_id == "voice-model"
            target = self.output_dir / "simulated-rvc.wav"
            sf.write(target, np.zeros(4800, dtype="float32"), 48000)
            return EngineResult(str(target), 48000, "simulated-applio", {"duration": 0.1})

    monkeypatch.setattr(app_module, "ApplioAdapter", FakeAdapter)
    audio = io.BytesIO()
    sf.write(audio, np.zeros(1600, dtype="float32"), 16000, format="WAV")
    client = TestClient(app_module.app)
    response = client.post(
        "/api/workflow/voice-conversion",
        data={"model_id": "voice-model", "pitch": "2", "f0_method": "rmvpe"},
        files={"input_audio": ("input.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["result"]["engine"] == "simulated-applio"


def test_separation_workflow_with_simulated_adapter(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    inbox.mkdir()
    outputs.mkdir()
    monkeypatch.setenv("DEMUCS_EXECUTABLE", "demucs-engine")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    app_module.jobs.clear()

    class FakeAdapter:
        def __init__(self, executable, output_dir):
            self.output_dir = output_dir

        async def run(self, request):
            vocals = self.output_dir / "vocals.wav"
            accompaniment = self.output_dir / "accompaniment.wav"
            sf.write(vocals, np.zeros(4410, dtype="float32"), 44100)
            sf.write(accompaniment, np.zeros(4410, dtype="float32"), 44100)
            return EngineResult(
                str(vocals),
                44100,
                "simulated-demucs",
                {"duration": 0.1},
                {"vocals": str(vocals), "accompaniment": str(accompaniment)},
            )

    monkeypatch.setattr(app_module, "DemucsAdapter", FakeAdapter)
    audio = io.BytesIO()
    sf.write(audio, np.zeros(4410, dtype="float32"), 44100, format="WAV")
    client = TestClient(app_module.app)
    response = client.post(
        "/api/workflow/separate",
        data={"model": "htdemucs"},
        files={"input_audio": ("song.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert set(job["artifacts"]) == {"vocals", "accompaniment"}
    assert all(client.get(url).status_code == 200 for url in job["artifacts"].values())


def test_mix_workflow_is_real_audio_processing(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    inbox.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    app_module.jobs.clear()
    vocals = io.BytesIO()
    accompaniment = io.BytesIO()
    sf.write(vocals, np.ones(8000, dtype="float32") * 0.6, 8000, format="WAV")
    sf.write(accompaniment, np.ones(4000, dtype="float32") * 0.6, 8000, format="WAV")
    client = TestClient(app_module.app)
    response = client.post(
        "/api/workflow/mix",
        data={"vocal_gain_db": "-3", "accompaniment_gain_db": "0"},
        files={
            "vocals": ("vocals.wav", vocals.getvalue(), "audio/wav"),
            "accompaniment": ("accompaniment.wav", accompaniment.getvalue(), "audio/wav"),
        },
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["result"]["duration"] == 1.0
    assert client.get(job["output"]).status_code == 200


def test_midi_inspection_api(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    events = b"\x00\x90\x3c\x64\x83\x60\x80\x3c\x00\x00\xff\x2f\x00"
    midi = (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", len(events))
        + events
    )
    client = TestClient(app_module.app)
    response = client.post("/api/midi/inspect", files={"file": ("melody.mid", midi, "audio/midi")})

    assert response.status_code == 200
    assert response.json()["status"] == "parsed_no_synthesis"
    assert response.json()["notes"][0]["pitch"] == 60


def test_singing_workflow_with_simulated_adapter(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    engine = tmp_path / "diffsinger"
    for directory in (inbox, outputs, engine):
        directory.mkdir()
    monkeypatch.setenv("DIFFSINGER_ROOT", str(engine))
    monkeypatch.setenv("DIFFSINGER_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    app_module.jobs.clear()
    app_module.cancel_flags.clear()

    class FakeAdapter:
        def __init__(self, diffsinger_root, python_executable, output_dir):
            self.output_dir = output_dir

        async def run(self, request):
            assert request.parameters["experiment"] == "licensed-singer"
            assert Path(request.parameters["ds_path"]).suffix == ".ds"
            target = self.output_dir / "simulated-singing.wav"
            sf.write(target, np.zeros(4410, dtype="float32"), 44100)
            return EngineResult(str(target), 44100, "simulated-diffsinger", {"duration": 0.1})

    monkeypatch.setattr(app_module, "DiffSingerAdapter", FakeAdapter, raising=False)
    client = TestClient(app_module.app)
    response = client.post(
        "/api/workflow/singing",
        data={"experiment": "licensed-singer", "speaker": "singer-a", "key": "1"},
        files={"ds_file": ("song.ds", b'[{"ph_seq":"a"}]', "application/json")},
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["result"]["engine"] == "simulated-diffsinger"
    assert client.get(job["output"]).status_code == 200
    assert client.get("/api/capabilities").json()["singing_synthesis"] == "configured_unverified"
