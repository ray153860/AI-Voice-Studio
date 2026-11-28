import asyncio
import io
import struct
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from voice_studio import app as app_module
from voice_studio.engines import EngineResult, TrainingResult


class ApiClient:
    def __init__(self, application):
        self.application = application

    def request(self, method, url, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.application)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


def test_home_exposes_diffsinger_submission_controls():
    response = ApiClient(app_module.app).get("/")

    assert response.status_code == 200
    assert 'id="singingDs"' in response.text
    assert 'id="singingExperiment"' in response.text
    assert 'id="singingLyrics"' in response.text
    assert "对齐歌词与 MIDI" in response.text
    assert "提交 DiffSinger 任务" in response.text
    assert 'id="datasetFiles"' in response.text
    assert 'id="trainingDataset"' in response.text
    assert "提交 RVC 训练任务" in response.text
    assert 'id="coverSong"' in response.text
    assert 'id="coverModel"' in response.text
    assert 'id="coverChunk"' in response.text
    assert 'id="coverOverlap"' in response.text
    assert "运行翻唱预设工作流" in response.text
    assert 'id="jobControlId"' in response.text
    assert "重新执行任务" in response.text
    assert 'id="audioProject"' in response.text
    assert 'id="ttsProject"' in response.text
    assert 'id="rvcProject"' in response.text
    assert 'id="coverProject"' in response.text
    assert 'id="separationProject"' in response.text
    assert 'id="singingProject"' in response.text
    assert 'id="trainingProject"' in response.text
    assert 'id="mixProject"' in response.text
    assert 'id="jobsProject"' in response.text
    assert '<select id="rvcModel"' in response.text
    assert '<select id="rvcIndex"' in response.text
    assert '<select id="coverModel"' in response.text
    assert '<select id="coverIndex"' in response.text
    assert '<select id="trainingDataset"' in response.text
    assert 'id="outputProject"' in response.text
    assert "查看项目输出" in response.text


def test_capabilities_show_training_dependencies_are_missing(monkeypatch):
    monkeypatch.delenv("APPLIO_ROOT", raising=False)
    monkeypatch.delenv("APPLIO_PYTHON", raising=False)
    capabilities = ApiClient(app_module.app).get("/api/capabilities").json()

    assert capabilities["model_training"] == "interface_only_dependencies_missing"


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

    client = ApiClient(app_module.app)
    created = client.post("/api/projects", json={"name": "Demo"})
    assert created.status_code == 200
    assert client.get("/api/projects").json()[0]["name"] == "Demo"
    project_id = created.json()["id"]

    buf = io.BytesIO()
    sf.write(buf, (np.sin(np.arange(8000) * .03) * .2).astype("float32"), 8000, format="WAV")
    response = client.post(
        "/api/workflow/audio",
        data={"project_id": project_id},
        files={"file": ("tone.wav", buf.getvalue(), "audio/wav")},
    )
    assert response.status_code == 200
    jid = response.json()["id"]
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["project_id"] == project_id
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


def test_jobs_can_be_filtered_by_project(tmp_path, monkeypatch):
    from voice_studio.store import Store

    store = Store(tmp_path / "studio.sqlite3")
    store.save_job({"id": "job-a", "status": "completed", "project_id": "project-a", "log": []})
    store.save_job({"id": "job-b", "status": "completed", "project_id": "project-b", "log": []})
    store.save_job({"id": "job-unassigned", "status": "completed", "project_id": "", "log": []})
    monkeypatch.setattr(app_module, "store", store)

    jobs = ApiClient(app_module.app).get("/api/jobs", params={"project_id": "project-a"}).json()

    assert [job["id"] for job in jobs] == ["job-a"]


def test_project_outputs_collect_downloadable_job_files(tmp_path, monkeypatch):
    from voice_studio.store import Store

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "vocals.wav").write_bytes(b"vocals")
    (outputs / "accompaniment.wav").write_bytes(b"accompaniment")
    store = Store(tmp_path / "studio.sqlite3")
    store.create_project("project-a", "Project A")
    store.save_job(
        {
            "id": "separation-job",
            "type": "vocal_separation",
            "status": "completed",
            "project_id": "project-a",
            "output": "/api/outputs/vocals.wav",
            "artifacts": {
                "vocals": "/api/outputs/vocals.wav",
                "accompaniment": "/api/outputs/accompaniment.wav",
                "external": "https://example.test/not-local.wav",
            },
            "log": [],
        }
    )
    store.save_job(
        {
            "id": "other-project-job",
            "type": "audio_mix",
            "status": "completed",
            "project_id": "project-b",
            "output": "/api/outputs/other.wav",
            "log": [],
        }
    )
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", store)

    response = ApiClient(app_module.app).get("/api/projects/project-a/outputs")

    assert response.status_code == 200
    assert response.json() == [
        {
            "job_id": "separation-job",
            "job_type": "vocal_separation",
            "label": "output",
            "url": "/api/outputs/vocals.wav",
        },
        {
            "job_id": "separation-job",
            "job_type": "vocal_separation",
            "label": "accompaniment",
            "url": "/api/outputs/accompaniment.wav",
        },
    ]


def test_project_outputs_omit_missing_or_unsafe_files(tmp_path, monkeypatch):
    from voice_studio.store import Store

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "available.wav").write_bytes(b"audio")
    store = Store(tmp_path / "studio.sqlite3")
    store.create_project("project-a", "Project A")
    store.save_job(
        {
            "id": "mixed-outputs",
            "type": "audio_mix",
            "status": "completed",
            "project_id": "project-a",
            "output": "/api/outputs/available.wav",
            "artifacts": {
                "missing": "/api/outputs/missing.wav",
                "unsafe": "/api/outputs/../private.wav",
            },
            "log": [],
        }
    )
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", store)

    response = ApiClient(app_module.app).get("/api/projects/project-a/outputs")

    assert response.status_code == 200
    assert [item["url"] for item in response.json()] == ["/api/outputs/available.wav"]


def test_project_outputs_reject_unknown_project(tmp_path, monkeypatch):
    from voice_studio.store import Store

    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))

    response = ApiClient(app_module.app).get("/api/projects/missing-project/outputs")

    assert response.status_code == 404
    assert response.json()["detail"] == "project not found"


def test_invalid_audio_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    client = ApiClient(app_module.app)
    assert client.post("/api/workflow/audio", files={"file": ("bad.wav", b"not audio")}).status_code == 400


def test_audio_workflow_rejects_unknown_project(tmp_path, monkeypatch):
    from voice_studio.store import Store

    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    audio = io.BytesIO()
    sf.write(audio, np.zeros(80, dtype="float32"), 8000, format="WAV")

    response = ApiClient(app_module.app).post(
        "/api/workflow/audio",
        data={"project_id": "missing-project"},
        files={"file": ("tone.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must reference an existing project"
    assert not list(tmp_path.glob("*.wav"))


def test_audio_analysis_rejects_payload_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/audio/analyze",
        files={"file": ("too-large.wav", b"123456789", "audio/wav")},
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_audio_analysis_removes_temporary_file_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    audio = io.BytesIO()
    sf.write(audio, np.zeros(800, dtype="float32"), 8000, format="WAV")

    response = ApiClient(app_module.app).post(
        "/api/audio/analyze",
        files={"file": ("tone.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["sample_rate"] == 8000
    assert not list(tmp_path.iterdir())


def test_audio_compare_cleans_first_file_when_second_exceeds_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/audio/compare",
        files={
            "before": ("before.wav", b"123", "audio/wav"),
            "after": ("after.wav", b"123456789", "audio/wav"),
        },
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_audio_compare_removes_temporary_files_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    before = io.BytesIO()
    after = io.BytesIO()
    sf.write(before, np.zeros(800, dtype="float32"), 8000, format="WAV")
    sf.write(after, np.ones(800, dtype="float32") * 0.1, 8000, format="WAV")

    response = ApiClient(app_module.app).post(
        "/api/audio/compare",
        files={
            "before": ("before.wav", before.getvalue(), "audio/wav"),
            "after": ("after.wav", after.getvalue(), "audio/wav"),
        },
    )

    assert response.status_code == 200
    assert not list(tmp_path.iterdir())


def test_audio_upload_rejects_payload_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8, raising=False)

    response = ApiClient(app_module.app).post(
        "/api/workflow/audio",
        files={"file": ("too-large.wav", b"123456789", "audio/wav")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "audio upload exceeds the 8 byte limit"
    assert not list(tmp_path.iterdir())


def test_cancel_queued_job(tmp_path, monkeypatch):
    from voice_studio.store import Store

    store = Store(tmp_path / "studio.sqlite3")
    monkeypatch.setattr(app_module, "store", store)
    app_module.jobs.clear()
    app_module.cancel_flags.clear()
    job = {"id": "queued-job", "status": "queued", "input": "tone.wav", "metadata": {}, "log": []}
    app_module.jobs[job["id"]] = job
    store.save_job(job)

    client = ApiClient(app_module.app)
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
    client = ApiClient(app_module.app)

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


def test_model_import_rejects_payload_over_limit(tmp_path, monkeypatch):
    from voice_studio.store import Store

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(app_module, "MODELS", models)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    monkeypatch.setattr(app_module, "MAX_MODEL_UPLOAD_BYTES", 8, raising=False)

    response = ApiClient(app_module.app).post(
        "/api/models",
        data={"model_type": "rvc", "license_name": "user-owned"},
        files={"file": ("voice.pth", b"123456789", "application/octet-stream")},
    )

    assert response.status_code == 413
    assert not list(models.iterdir())


def test_tts_requires_configured_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("GPT_SOVITS_URL", raising=False)
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    client = ApiClient(app_module.app)
    response = client.post(
        "/api/workflow/tts",
        data={"text": "测试", "text_lang": "zh", "prompt_lang": "zh"},
        files={"reference_audio": ("reference.wav", b"not read because engine is unavailable")},
    )
    assert response.status_code == 503
    assert client.get("/api/engines").json()[0]["configured"] is False


def test_tts_rejects_reference_audio_over_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("GPT_SOVITS_URL", "http://engine.test:9880")
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/workflow/tts",
        data={"text": "测试"},
        files={"reference_audio": ("reference.wav", b"123456789", "audio/wav")},
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_tts_workflow_with_simulated_adapter(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    inbox.mkdir()
    outputs.mkdir()
    monkeypatch.setenv("GPT_SOVITS_URL", "http://engine.test:9880")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    store = Store(tmp_path / "studio.sqlite3")
    store.create_project("tts-project", "TTS Project")
    monkeypatch.setattr(app_module, "store", store)
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
    client = ApiClient(app_module.app)
    response = client.post(
        "/api/workflow/tts",
        data={"text": "测试", "text_lang": "zh", "prompt_lang": "zh", "project_id": "tts-project"},
        files={"reference_audio": ("reference.wav", reference.getvalue(), "audio/wav")},
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["project_id"] == "tts-project"
    assert job["result"]["engine"] == "simulated-gpt-sovits"
    assert client.get(job["output"]).status_code == 200

    retried = client.post(f"/api/jobs/{job['id']}/retry")
    assert retried.status_code == 200
    retried_job = client.get(f"/api/jobs/{retried.json()['id']}").json()
    assert retried_job["status"] == "completed"
    assert retried_job["id"] != job["id"]
    assert retried_job["project_id"] == "tts-project"
    assert retried_job["log"][0] == f"retried from {job['id']}"
    assert retried_job["result"]["engine"] == "simulated-gpt-sovits"


def test_tts_workflow_rejects_unknown_project_before_upload(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    inbox.mkdir()
    monkeypatch.setenv("GPT_SOVITS_URL", "http://engine.test:9880")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    reference = io.BytesIO()
    sf.write(reference, np.zeros(800, dtype="float32"), 8000, format="WAV")

    response = ApiClient(app_module.app).post(
        "/api/workflow/tts",
        data={"text": "test", "project_id": "missing-project"},
        files={"reference_audio": ("reference.wav", reference.getvalue(), "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must reference an existing project"
    assert not list(inbox.iterdir())


def test_voice_conversion_requires_configured_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("APPLIO_ROOT", raising=False)
    monkeypatch.delenv("APPLIO_PYTHON", raising=False)
    client = ApiClient(app_module.app)
    response = client.post(
        "/api/workflow/voice-conversion",
        data={"model_id": "missing"},
        files={"input_audio": ("input.wav", b"unused")},
    )
    assert response.status_code == 503


def test_voice_conversion_rejects_input_audio_over_limit(tmp_path, monkeypatch):
    from voice_studio.store import Store

    engine = tmp_path / "applio"
    engine.mkdir()
    model_path = tmp_path / "voice.pth"
    model_path.write_bytes(b"model")
    store = Store(tmp_path / "studio.sqlite3")
    store.save_model({"id": "voice-model", "type": "rvc", "name": "voice.pth", "path": str(model_path)})
    inbox = tmp_path / "audio"
    inbox.mkdir()
    monkeypatch.setenv("APPLIO_ROOT", str(engine))
    monkeypatch.setenv("APPLIO_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "store", store)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/workflow/voice-conversion",
        data={"model_id": "voice-model"},
        files={"input_audio": ("input.wav", b"123456789", "audio/wav")},
    )

    assert response.status_code == 413
    assert not list(inbox.iterdir())


def test_voice_conversion_rejects_unknown_project_before_upload(tmp_path, monkeypatch):
    from voice_studio.store import Store

    engine = tmp_path / "applio"
    inbox = tmp_path / "audio"
    engine.mkdir()
    inbox.mkdir()
    monkeypatch.setenv("APPLIO_ROOT", str(engine))
    monkeypatch.setenv("APPLIO_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    audio = io.BytesIO()
    sf.write(audio, np.zeros(800, dtype="float32"), 8000, format="WAV")

    response = ApiClient(app_module.app).post(
        "/api/workflow/voice-conversion",
        data={"model_id": "missing-model", "project_id": "missing-project"},
        files={"input_audio": ("input.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must reference an existing project"
    assert not list(inbox.iterdir())


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
    store.create_project("rvc-project", "RVC Project")
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
    client = ApiClient(app_module.app)
    response = client.post(
        "/api/workflow/voice-conversion",
        data={"model_id": "voice-model", "pitch": "2", "f0_method": "rmvpe", "project_id": "rvc-project"},
        files={"input_audio": ("input.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["project_id"] == "rvc-project"
    assert job["result"]["engine"] == "simulated-applio"

    retried = client.post(f"/api/jobs/{job['id']}/retry")
    assert retried.status_code == 200
    retried_job = client.get(f"/api/jobs/{retried.json()['id']}").json()
    assert retried_job["status"] == "completed"
    assert retried_job["project_id"] == "rvc-project"
    assert retried_job["log"][0] == f"retried from {job['id']}"
    assert retried_job["result"]["engine"] == "simulated-applio"


def test_separation_workflow_with_simulated_adapter(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    inbox.mkdir()
    outputs.mkdir()
    monkeypatch.setenv("DEMUCS_EXECUTABLE", "demucs-engine")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    store = Store(tmp_path / "studio.sqlite3")
    store.create_project("separation-project", "Separation Project")
    monkeypatch.setattr(app_module, "store", store)
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
    client = ApiClient(app_module.app)
    response = client.post(
        "/api/workflow/separate",
        data={"model": "htdemucs", "project_id": "separation-project"},
        files={"input_audio": ("song.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["project_id"] == "separation-project"
    assert set(job["artifacts"]) == {"vocals", "accompaniment"}
    assert all(client.get(url).status_code == 200 for url in job["artifacts"].values())

    retried = client.post(f"/api/jobs/{job['id']}/retry")
    assert retried.status_code == 200
    retried_job = client.get(f"/api/jobs/{retried.json()['id']}").json()
    assert retried_job["status"] == "completed"
    assert retried_job["project_id"] == "separation-project"
    assert retried_job["log"][0] == f"retried from {job['id']}"
    assert retried_job["result"]["engine"] == "simulated-demucs"


def test_separation_rejects_input_audio_over_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMUCS_EXECUTABLE", "demucs-engine")
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/workflow/separate",
        files={"input_audio": ("song.wav", b"123456789", "audio/wav")},
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_separation_rejects_unknown_project_before_upload(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    inbox.mkdir()
    monkeypatch.setenv("DEMUCS_EXECUTABLE", "demucs-engine")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    audio = io.BytesIO()
    sf.write(audio, np.zeros(800, dtype="float32"), 8000, format="WAV")

    response = ApiClient(app_module.app).post(
        "/api/workflow/separate",
        data={"project_id": "missing-project"},
        files={"input_audio": ("song.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must reference an existing project"
    assert not list(inbox.iterdir())


def test_mix_workflow_is_real_audio_processing(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    inbox.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    store = Store(tmp_path / "studio.sqlite3")
    store.create_project("mix-project", "Mix Project")
    monkeypatch.setattr(app_module, "store", store)
    app_module.jobs.clear()
    vocals = io.BytesIO()
    accompaniment = io.BytesIO()
    sf.write(vocals, np.ones(8000, dtype="float32") * 0.6, 8000, format="WAV")
    sf.write(accompaniment, np.ones(4000, dtype="float32") * 0.6, 8000, format="WAV")
    client = ApiClient(app_module.app)
    response = client.post(
        "/api/workflow/mix",
        data={"vocal_gain_db": "-3", "accompaniment_gain_db": "0", "project_id": "mix-project"},
        files={
            "vocals": ("vocals.wav", vocals.getvalue(), "audio/wav"),
            "accompaniment": ("accompaniment.wav", accompaniment.getvalue(), "audio/wav"),
        },
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["project_id"] == "mix-project"
    assert job["result"]["duration"] == 1.0
    assert client.get(job["output"]).status_code == 200

    retried = client.post(f"/api/jobs/{job['id']}/retry")
    assert retried.status_code == 200
    retried_job = client.get(f"/api/jobs/{retried.json()['id']}").json()
    assert retried_job["status"] == "completed"
    assert retried_job["project_id"] == "mix-project"
    assert retried_job["log"][0] == f"retried from {job['id']}"
    assert retried_job["result"]["duration"] == 1.0


def test_mix_rejects_unknown_project_before_upload(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    inbox.mkdir()
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    vocals = io.BytesIO()
    accompaniment = io.BytesIO()
    sf.write(vocals, np.zeros(800, dtype="float32"), 8000, format="WAV")
    sf.write(accompaniment, np.zeros(800, dtype="float32"), 8000, format="WAV")

    response = ApiClient(app_module.app).post(
        "/api/workflow/mix",
        data={"project_id": "missing-project"},
        files={
            "vocals": ("vocals.wav", vocals.getvalue(), "audio/wav"),
            "accompaniment": ("accompaniment.wav", accompaniment.getvalue(), "audio/wav"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must reference an existing project"
    assert not list(inbox.iterdir())


def test_mix_rejects_vocal_audio_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/workflow/mix",
        files={
            "vocals": ("vocals.wav", b"123456789", "audio/wav"),
            "accompaniment": ("accompaniment.wav", b"123", "audio/wav"),
        },
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_mix_cleans_vocal_when_accompaniment_exceeds_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/workflow/mix",
        files={
            "vocals": ("vocals.wav", b"123", "audio/wav"),
            "accompaniment": ("accompaniment.wav", b"123456789", "audio/wav"),
        },
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_mix_runner_honors_cancellation_before_processing(tmp_path, monkeypatch):
    from voice_studio.store import Store

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    store = Store(tmp_path / "studio.sqlite3")
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", store)
    app_module.jobs.clear()
    app_module.cancel_flags.clear()
    vocals = tmp_path / "vocals.wav"
    accompaniment = tmp_path / "accompaniment.wav"
    sf.write(vocals, np.ones(800, dtype="float32") * 0.2, 8000)
    sf.write(accompaniment, np.ones(800, dtype="float32") * 0.2, 8000)
    job = {"id": "mix-cancelled", "type": "audio_mix", "status": "queued", "input": [], "metadata": {}, "log": []}
    app_module.jobs[job["id"]] = job
    store.save_job(job)
    app_module.cancel_flags[job["id"]] = True

    asyncio.run(app_module.run_mix_job(job["id"], vocals, accompaniment, 0, 0))

    assert store.get_job(job["id"])["status"] == "cancelled"
    assert not (outputs / f"mix-{job['id']}.wav").exists()


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
    client = ApiClient(app_module.app)
    response = client.post("/api/midi/inspect", files={"file": ("melody.mid", midi, "audio/midi")})

    assert response.status_code == 200
    assert response.json()["status"] == "parsed_no_synthesis"
    assert response.json()["notes"][0]["pitch"] == 60
    assert not list(tmp_path.iterdir())


def test_midi_inspection_rejects_payload_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_DOCUMENT_UPLOAD_BYTES", 8, raising=False)

    response = ApiClient(app_module.app).post(
        "/api/midi/inspect",
        files={"file": ("melody.mid", b"123456789", "audio/midi")},
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_midi_lyrics_alignment_api(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    events = b"\x00\x90\x3c\x64\x83\x60\x80\x3c\x00\x00\xff\x2f\x00"
    midi = (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", len(events))
        + events
    )
    client = ApiClient(app_module.app)

    response = client.post(
        "/api/singing/align",
        data={"lyrics": "你"},
        files={"midi_file": ("melody.mid", midi, "audio/midi")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "aligned_no_phonemization_no_synthesis"
    assert response.json()["score"][0]["lyric"] == "你"
    assert response.json()["score"][0]["pitch"] == 60
    assert not list(tmp_path.iterdir())


def test_midi_lyrics_alignment_rejects_payload_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INBOX", tmp_path)
    monkeypatch.setattr(app_module, "MAX_DOCUMENT_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/singing/align",
        data={"lyrics": "你"},
        files={"midi_file": ("melody.mid", b"123456789", "audio/midi")},
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_singing_workflow_rejects_ds_file_over_limit(tmp_path, monkeypatch):
    engine = tmp_path / "diffsinger"
    engine.mkdir()
    inbox = tmp_path / "audio"
    inbox.mkdir()
    monkeypatch.setenv("DIFFSINGER_ROOT", str(engine))
    monkeypatch.setenv("DIFFSINGER_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "MAX_DOCUMENT_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/workflow/singing",
        data={"experiment": "licensed-singer"},
        files={"ds_file": ("song.ds", b"123456789", "application/json")},
    )

    assert response.status_code == 413
    assert not list(inbox.iterdir())


def test_singing_workflow_rejects_unknown_project_before_upload(tmp_path, monkeypatch):
    from voice_studio.store import Store

    engine = tmp_path / "diffsinger"
    inbox = tmp_path / "audio"
    engine.mkdir()
    inbox.mkdir()
    monkeypatch.setenv("DIFFSINGER_ROOT", str(engine))
    monkeypatch.setenv("DIFFSINGER_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))

    response = ApiClient(app_module.app).post(
        "/api/workflow/singing",
        data={"experiment": "licensed-singer", "project_id": "missing-project"},
        files={"ds_file": ("song.ds", b'[{"ph_seq":"a"}]', "application/json")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must reference an existing project"
    assert not list(inbox.iterdir())


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
    store = Store(tmp_path / "studio.sqlite3")
    store.create_project("singing-project", "Singing Project")
    monkeypatch.setattr(app_module, "store", store)
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
    client = ApiClient(app_module.app)
    response = client.post(
        "/api/workflow/singing",
        data={"experiment": "licensed-singer", "speaker": "singer-a", "key": "1", "project_id": "singing-project"},
        files={"ds_file": ("song.ds", b'[{"ph_seq":"a"}]', "application/json")},
    )

    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["project_id"] == "singing-project"
    assert job["result"]["engine"] == "simulated-diffsinger"
    assert client.get(job["output"]).status_code == 200
    assert client.get("/api/capabilities").json()["singing_synthesis"] == "configured_unverified"

    retried = client.post(f"/api/jobs/{job['id']}/retry")
    assert retried.status_code == 200
    retried_job = client.get(f"/api/jobs/{retried.json()['id']}").json()
    assert retried_job["status"] == "completed"
    assert retried_job["project_id"] == "singing-project"
    assert retried_job["log"][0] == f"retried from {job['id']}"
    assert retried_job["result"]["engine"] == "simulated-diffsinger"


def test_dataset_import_and_training_workflow_with_simulated_adapter(tmp_path, monkeypatch):
    from voice_studio.store import Store

    datasets = tmp_path / "datasets"
    models = tmp_path / "models"
    engine = tmp_path / "applio"
    for directory in (datasets, models, engine):
        directory.mkdir()
    monkeypatch.setenv("APPLIO_ROOT", str(engine))
    monkeypatch.setenv("APPLIO_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "DATASETS", datasets, raising=False)
    monkeypatch.setattr(app_module, "MODELS", models)
    store = Store(tmp_path / "studio.sqlite3")
    store.create_project("training-project", "Training Project")
    monkeypatch.setattr(app_module, "store", store)
    app_module.jobs.clear()
    tone = io.BytesIO()
    sf.write(tone, np.zeros(1600, dtype="float32"), 16000, format="WAV")
    client = ApiClient(app_module.app)
    imported = client.post(
        "/api/datasets",
        data={"name": "Authorized clips", "license_name": "owned-recordings"},
        files=[
            ("files", ("clip-a.wav", tone.getvalue(), "audio/wav")),
            ("files", ("clip-b.wav", tone.getvalue(), "audio/wav")),
        ],
    )

    assert imported.status_code == 200
    dataset = imported.json()
    assert dataset["files"] == 2
    assert client.get("/api/datasets").json()[0]["license"] == "owned-recordings"

    class FakeTrainingAdapter:
        def __init__(self, applio_root, python_executable):
            self.applio_root = applio_root

        async def run(self, request):
            assert request.model_name == "DemoVoice"
            assert Path(request.dataset_path).is_dir()
            output = self.applio_root / "logs" / "DemoVoice"
            output.mkdir(parents=True, exist_ok=True)
            model = output / "DemoVoice.pth"
            index = output / "DemoVoice.index"
            model.write_bytes(b"model")
            index.write_bytes(b"index")
            return TrainingResult("simulated-applio-training", {"model": str(model), "index": str(index)})

    monkeypatch.setattr(app_module, "ApplioTrainingAdapter", FakeTrainingAdapter, raising=False)
    started = client.post(
        "/api/workflow/train-rvc",
        data={
            "dataset_id": dataset["id"],
            "model_name": "DemoVoice",
            "total_epoch": "20",
            "project_id": "training-project",
        },
    )

    assert started.status_code == 200
    job = client.get(f"/api/jobs/{started.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["project_id"] == "training-project"
    assert job["result"]["engine"] == "simulated-applio-training"
    trained = [model for model in client.get("/api/models").json() if model["status"] == "trained_unverified"]
    assert {Path(model["name"]).suffix for model in trained} == {".pth", ".index"}

    retried = client.post(f"/api/jobs/{job['id']}/retry")
    assert retried.status_code == 200
    retried_job = client.get(f"/api/jobs/{retried.json()['id']}").json()
    assert retried_job["status"] == "completed"
    assert retried_job["project_id"] == "training-project"
    assert retried_job["log"][0] == f"retried from {job['id']}"
    assert retried_job["result"]["engine"] == "simulated-applio-training"


def test_training_workflow_rejects_unknown_project(tmp_path, monkeypatch):
    from voice_studio.store import Store

    engine = tmp_path / "applio"
    engine.mkdir()
    monkeypatch.setenv("APPLIO_ROOT", str(engine))
    monkeypatch.setenv("APPLIO_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))

    response = ApiClient(app_module.app).post(
        "/api/workflow/train-rvc",
        data={"dataset_id": "missing-dataset", "model_name": "DemoVoice", "project_id": "missing-project"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must reference an existing project"


def test_dataset_import_rejects_audio_over_limit_and_cleans_directory(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    monkeypatch.setattr(app_module, "DATASETS", datasets)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/datasets",
        data={"name": "Authorized clips", "license_name": "owned-recordings"},
        files=[("files", ("clip.wav", b"123456789", "audio/wav"))],
    )

    assert response.status_code == 413
    assert not list(datasets.iterdir())


def test_dataset_import_rejects_aggregate_over_limit_and_cleans_files(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    tiny_wav = io.BytesIO()
    sf.write(tiny_wav, np.zeros(1, dtype="float32"), 8000, format="WAV")
    payload = tiny_wav.getvalue()
    monkeypatch.setattr(app_module, "DATASETS", datasets)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", len(payload) + 1)
    monkeypatch.setattr(app_module, "MAX_DATASET_UPLOAD_BYTES", len(payload) + 1, raising=False)

    response = ApiClient(app_module.app).post(
        "/api/datasets",
        data={"name": "Authorized clips", "license_name": "owned-recordings"},
        files=[
            ("files", ("clip-a.wav", payload, "audio/wav")),
            ("files", ("clip-b.wav", payload, "audio/wav")),
        ],
    )

    assert response.status_code == 413
    assert not list(datasets.iterdir())


def test_cover_workflow_with_simulated_ai_adapters(tmp_path, monkeypatch):
    from voice_studio.store import Store

    inbox = tmp_path / "audio"
    outputs = tmp_path / "outputs"
    engine = tmp_path / "applio"
    for directory in (inbox, outputs, engine):
        directory.mkdir()
    model_path = tmp_path / "voice.pth"
    model_path.write_bytes(b"model")
    store = Store(tmp_path / "studio.sqlite3")
    store.create_project("cover-project", "Cover Project")
    store.save_model(
        {"id": "cover-voice", "type": "rvc", "name": "voice.pth", "path": str(model_path), "status": "registered_unverified"}
    )
    monkeypatch.setenv("DEMUCS_EXECUTABLE", "demucs-engine")
    monkeypatch.setenv("APPLIO_ROOT", str(engine))
    monkeypatch.setenv("APPLIO_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "OUT", outputs)
    monkeypatch.setattr(app_module, "store", store)
    app_module.jobs.clear()

    class FakeSeparator:
        def __init__(self, executable, output_dir):
            self.output_dir = output_dir

        async def run(self, request):
            vocals = self.output_dir / "cover-vocals.wav"
            accompaniment = self.output_dir / "cover-accompaniment.wav"
            sf.write(vocals, np.sin(np.arange(8000) * 0.1) * 0.2, 8000)
            sf.write(accompaniment, np.sin(np.arange(8000) * 0.05) * 0.1, 8000)
            return EngineResult(
                str(vocals),
                8000,
                "simulated-demucs",
                artifacts={"vocals": str(vocals), "accompaniment": str(accompaniment)},
            )

    class FakeConverter:
        def __init__(self, applio_root, python_executable, output_dir):
            self.output_dir = output_dir
            self.calls = 0

        async def run(self, request):
            assert request.parameters["model_path"] == str(model_path)
            self.calls += 1
            chunk, sample_rate = sf.read(request.input_audio, dtype="float32")
            converted = self.output_dir / f"cover-converted-{self.calls}.wav"
            sf.write(converted, chunk * 1.1, sample_rate)
            return EngineResult(str(converted), 8000, "simulated-applio")

    monkeypatch.setattr(app_module, "DemucsAdapter", FakeSeparator)
    monkeypatch.setattr(app_module, "ApplioAdapter", FakeConverter)
    song = io.BytesIO()
    sf.write(song, np.zeros(8000, dtype="float32"), 8000, format="WAV")
    client = ApiClient(app_module.app)
    started = client.post(
        "/api/workflow/cover",
        data={
            "model_id": "cover-voice",
            "pitch": "2",
            "chunk_seconds": "0.5",
            "overlap_seconds": "0.1",
            "project_id": "cover-project",
        },
        files={"song": ("song.wav", song.getvalue(), "audio/wav")},
    )

    assert started.status_code == 200
    job = client.get(f"/api/jobs/{started.json()['id']}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["progress_stage"] == "mix"
    assert job["project_id"] == "cover-project"
    assert job["result"]["stages"] == ["separation", "voice_conversion", "quality_comparison", "mix"]
    assert job["result"]["engines"] == {"separation": "simulated-demucs", "voice_conversion": "simulated-applio"}
    assert job["result"]["chunking"]["chunk_count"] == 3
    assert client.get(job["output"]).status_code == 200

    retried = client.post(f"/api/jobs/{job['id']}/retry")
    assert retried.status_code == 200
    retried_job = client.get(f"/api/jobs/{retried.json()['id']}").json()
    assert retried_job["status"] == "completed"
    assert retried_job["project_id"] == "cover-project"
    assert retried_job["log"][0] == f"retried from {job['id']}"
    assert retried_job["result"]["stages"][-1] == "mix"


def test_cover_rejects_song_audio_over_limit(tmp_path, monkeypatch):
    from voice_studio.store import Store

    engine = tmp_path / "applio"
    engine.mkdir()
    model_path = tmp_path / "voice.pth"
    model_path.write_bytes(b"model")
    store = Store(tmp_path / "studio.sqlite3")
    store.save_model({"id": "cover-voice", "type": "rvc", "name": "voice.pth", "path": str(model_path)})
    inbox = tmp_path / "audio"
    inbox.mkdir()
    monkeypatch.setenv("DEMUCS_EXECUTABLE", "demucs-engine")
    monkeypatch.setenv("APPLIO_ROOT", str(engine))
    monkeypatch.setenv("APPLIO_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "store", store)
    monkeypatch.setattr(app_module, "MAX_AUDIO_UPLOAD_BYTES", 8)

    response = ApiClient(app_module.app).post(
        "/api/workflow/cover",
        data={"model_id": "cover-voice"},
        files={"song": ("song.wav", b"123456789", "audio/wav")},
    )

    assert response.status_code == 413
    assert not list(inbox.iterdir())


def test_cover_rejects_unknown_project_before_upload(tmp_path, monkeypatch):
    from voice_studio.store import Store

    engine = tmp_path / "applio"
    inbox = tmp_path / "audio"
    engine.mkdir()
    inbox.mkdir()
    monkeypatch.setenv("DEMUCS_EXECUTABLE", "demucs-engine")
    monkeypatch.setenv("APPLIO_ROOT", str(engine))
    monkeypatch.setenv("APPLIO_PYTHON", "engine-python")
    monkeypatch.setattr(app_module, "INBOX", inbox)
    monkeypatch.setattr(app_module, "store", Store(tmp_path / "studio.sqlite3"))
    song = io.BytesIO()
    sf.write(song, np.zeros(800, dtype="float32"), 8000, format="WAV")

    response = ApiClient(app_module.app).post(
        "/api/workflow/cover",
        data={"model_id": "missing-model", "project_id": "missing-project"},
        files={"song": ("song.wav", song.getvalue(), "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must reference an existing project"
    assert not list(inbox.iterdir())
