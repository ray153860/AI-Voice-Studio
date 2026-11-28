import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path

import httpx
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .adapters.applio import ApplioAdapter
from .adapters.applio_training import ApplioTrainingAdapter
from .adapters.demucs import DemucsAdapter
from .adapters.diffsinger import DiffSingerAdapter
from .adapters.gpt_sovits import GptSovitsAdapter
from .audio import inspect_audio, mix_audio, process_audio
from .engines import EngineRequest, TrainingRequest
from .midi import align_lyrics_to_notes, parse_midi
from .models import sha256_file, validate_model_filename
from .quality import analyze_audio, compare_audio
from .store import Store
from .ui import PAGE
from .workflows import run_cover_workflow

ROOT = Path(os.environ.get("AI_VOICE_STUDIO_DATA_DIR", Path.cwd() / "work" / "runtime")).resolve()
INBOX, OUT, MODELS, DATASETS = ROOT / "audio", ROOT / "outputs", ROOT / "models", ROOT / "datasets"
MAX_AUDIO_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_DATASET_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
MAX_DOCUMENT_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_MODEL_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
for p in (INBOX, OUT, MODELS, DATASETS): p.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="AI Voice Studio", version="0.1.0")
jobs = {}
cancel_flags = {}
store = Store(ROOT / "studio.sqlite3")
store.recover_incomplete_jobs()


async def save_upload_limited(upload: UploadFile, target: Path, limit: int, label: str) -> int:
    size = 0
    try:
        with target.open("wb") as stream:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"{label} upload exceeds the {limit} byte limit")
                stream.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return size


@app.get("/", response_class=HTMLResponse)
async def home(): return PAGE
@app.get("/api/projects")
async def list_projects(): return store.list_projects()
@app.post("/api/projects")
async def create_project(payload: dict):
    name=str(payload.get("name", "")).strip()
    if not name or len(name)>100: raise HTTPException(400,"project name must be 1-100 characters")
    pid=uuid.uuid4().hex
    store.create_project(pid,name)
    return {"id":pid,"name":name}
@app.get("/api/projects/{pid}/outputs")
async def list_project_outputs(pid: str):
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    outputs = []
    for job in store.list_jobs(pid):
        seen = set()
        candidates = [("output", job.get("output"))]
        artifacts = job.get("artifacts")
        if isinstance(artifacts, dict):
            candidates.extend(artifacts.items())
        for label, url in candidates:
            if not isinstance(url, str) or not url.startswith("/api/outputs/") or url in seen:
                continue
            seen.add(url)
            outputs.append({"job_id": job["id"], "job_type": job.get("type", ""), "label": label, "url": url})
    return outputs
@app.get("/api/jobs")
async def list_jobs(project_id: str = ""): return store.list_jobs(project_id)
@app.post("/api/midi/inspect")
async def inspect_midi(file: UploadFile):
    safe = Path(file.filename or "melody.mid").name
    if Path(safe).suffix.lower() not in (".mid", ".midi"):
        raise HTTPException(400, "MIDI file must use .mid or .midi")
    target = INBOX / f"midi-{uuid.uuid4().hex}_{safe}"
    await save_upload_limited(file, target, MAX_DOCUMENT_UPLOAD_BYTES, "MIDI")
    try:
        result = parse_midi(target)
    except (ValueError, OSError) as error:
        target.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    target.unlink(missing_ok=True)
    return {"file": safe, "status": "parsed_no_synthesis", **result}


@app.post("/api/singing/align")
async def align_singing_score(midi_file: UploadFile, lyrics: str = Form()):
    safe = Path(midi_file.filename or "melody.mid").name
    if Path(safe).suffix.lower() not in (".mid", ".midi"):
        raise HTTPException(400, "MIDI file must use .mid or .midi")
    target = INBOX / f"singing-{uuid.uuid4().hex}_{safe}"
    await save_upload_limited(midi_file, target, MAX_DOCUMENT_UPLOAD_BYTES, "MIDI")
    try:
        parsed = parse_midi(target)
        score = align_lyrics_to_notes(parsed["notes"], lyrics)
    except (ValueError, OSError) as error:
        target.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    target.unlink(missing_ok=True)
    return {
        "file": safe,
        "status": "aligned_no_phonemization_no_synthesis",
        "ticks_per_beat": parsed["ticks_per_beat"],
        "tempos": parsed["tempos"],
        "rests": parsed["rests"],
        "score": score,
    }
@app.post("/api/audio/analyze")
async def audio_quality(file: UploadFile):
    safe = Path(file.filename or "audio.wav").name
    target = INBOX / f"analysis-{uuid.uuid4().hex}_{safe}"
    await save_upload_limited(file, target, MAX_AUDIO_UPLOAD_BYTES, "audio")
    try:
        result = {"file": safe, **analyze_audio(target)}
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        target.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    target.unlink(missing_ok=True)
    return result
@app.post("/api/audio/compare")
async def compare_audio_quality(before: UploadFile, after: UploadFile):
    comparison_id = uuid.uuid4().hex
    before_path = INBOX / f"comparison-{comparison_id}-before.wav"
    after_path = INBOX / f"comparison-{comparison_id}-after.wav"
    try:
        await save_upload_limited(before, before_path, MAX_AUDIO_UPLOAD_BYTES, "audio")
        await save_upload_limited(after, after_path, MAX_AUDIO_UPLOAD_BYTES, "audio")
    except HTTPException:
        before_path.unlink(missing_ok=True)
        after_path.unlink(missing_ok=True)
        raise
    try:
        result = compare_audio(before_path, after_path)
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        before_path.unlink(missing_ok=True)
        after_path.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    before_path.unlink(missing_ok=True)
    after_path.unlink(missing_ok=True)
    return result
@app.get("/api/capabilities")
async def capabilities():
    tts_status = "configured_unverified" if os.environ.get("GPT_SOVITS_URL") else "interface_only_dependencies_missing"
    rvc_status = "configured_unverified" if os.environ.get("APPLIO_ROOT") and os.environ.get("APPLIO_PYTHON") else "interface_only_dependencies_missing"
    training_status = rvc_status
    separation_status = "configured_unverified" if os.environ.get("DEMUCS_EXECUTABLE") else "interface_only_dependencies_missing"
    singing_status = "configured_unverified" if os.environ.get("DIFFSINGER_ROOT") and os.environ.get("DIFFSINGER_PYTHON") else "interface_only_dependencies_missing"
    return {
        "audio_processing": "verified_real_processing",
        "tts": tts_status,
        "voice_conversion": rvc_status,
        "vocal_separation": separation_status,
        "singing_synthesis": singing_status,
        "model_training": training_status,
    }
@app.get("/api/engines")
async def engines():
    return [
        {"id": "gpt-sovits", "capability": "tts", "configured": bool(os.environ.get("GPT_SOVITS_URL")), "verification": "not_run"},
        {"id": "applio", "capability": "voice_conversion", "configured": bool(os.environ.get("APPLIO_ROOT") and os.environ.get("APPLIO_PYTHON")), "verification": "not_run"},
        {"id": "demucs", "capability": "vocal_separation", "configured": bool(os.environ.get("DEMUCS_EXECUTABLE")), "verification": "not_run"},
        {"id": "diffsinger", "capability": "singing_synthesis", "configured": bool(os.environ.get("DIFFSINGER_ROOT") and os.environ.get("DIFFSINGER_PYTHON")), "verification": "not_run"},
    ]
@app.get("/api/models")
async def list_models(): return store.list_models()


@app.get("/api/datasets")
async def list_datasets():
    return store.list_datasets()


@app.post("/api/datasets")
async def import_dataset(files: list[UploadFile], name: str = Form(), license_name: str = Form()):
    if not name.strip() or len(name.strip()) > 100:
        raise HTTPException(400, "dataset name must be 1-100 characters")
    if not license_name.strip():
        raise HTTPException(400, "dataset license or authorization is required")
    if not 1 <= len(files) <= 1000:
        raise HTTPException(400, "dataset must contain 1-1000 audio files")
    dataset_id = uuid.uuid4().hex
    target_dir = DATASETS / dataset_id
    target_dir.mkdir(parents=True, exist_ok=False)
    saved = []
    metadata = []
    total_size = 0
    try:
        for index, upload in enumerate(files):
            safe = Path(upload.filename or f"audio-{index}.wav").name
            target = target_dir / f"{index:04d}_{safe}"
            size = await save_upload_limited(upload, target, MAX_AUDIO_UPLOAD_BYTES, "audio")
            saved.append(target)
            total_size += size
            if total_size > MAX_DATASET_UPLOAD_BYTES:
                raise HTTPException(413, f"dataset upload exceeds the {MAX_DATASET_UPLOAD_BYTES} byte limit")
            metadata.append(inspect_audio(target))
    except HTTPException:
        for target in saved:
            target.unlink(missing_ok=True)
        target_dir.rmdir()
        raise
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        for target in saved:
            target.unlink(missing_ok=True)
        target_dir.rmdir()
        raise HTTPException(400, f"invalid training audio: {error}") from error
    dataset = {
        "id": dataset_id,
        "name": name.strip(),
        "license": license_name.strip(),
        "files": len(saved),
        "duration": sum(item["duration"] for item in metadata),
        "path": str(target_dir),
        "status": "registered_audio_validated_training_unverified",
    }
    store.save_dataset(dataset)
    return dataset


@app.post("/api/models")
async def import_model(
    file: UploadFile,
    model_type: str = Form(),
    license_name: str = Form(),
    source_url: str = Form(default=""),
):
    try:
        safe = validate_model_filename(file.filename or "", model_type)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not license_name.strip():
        raise HTTPException(400, "model license is required")
    model_id = uuid.uuid4().hex
    target_dir = MODELS / model_id
    target_dir.mkdir(parents=True)
    target = target_dir / safe
    try:
        size = await save_upload_limited(file, target, MAX_MODEL_UPLOAD_BYTES, "model")
    except HTTPException:
        target_dir.rmdir()
        raise
    if size == 0:
        target.unlink(missing_ok=True)
        target_dir.rmdir()
        raise HTTPException(400, "model file is empty")
    model = {
        "id": model_id,
        "name": safe,
        "type": model_type,
        "size": size,
        "sha256": sha256_file(target),
        "license": license_name.strip(),
        "source_url": source_url.strip(),
        "status": "registered_unverified",
        "path": str(target),
    }
    store.save_model(model)
    return model
@app.post("/api/workflow/audio")
async def workflow(file: UploadFile, bg: BackgroundTasks, project_id: str = Form(default="")):
    if project_id and not store.get_project(project_id):
        raise HTTPException(400, "project_id must reference an existing project")
    jid=uuid.uuid4().hex
    safe=Path(file.filename or "audio.wav").name
    src=INBOX/f"{jid}_{safe}"
    await save_upload_limited(file, src, MAX_AUDIO_UPLOAD_BYTES, "audio")
    try: meta=inspect_audio(src)
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as e: src.unlink(missing_ok=True); raise HTTPException(400,str(e)) from e
    jobs[jid]={"id":jid,"type":"audio_processing","status":"queued","input":safe,"project_id":project_id,"metadata":meta,"log":[]}
    store.save_job(jobs[jid])
    bg.add_task(run_job,jid,src)
    return jobs[jid]
@app.post("/api/workflow/mix")
async def mix_workflow(
    bg: BackgroundTasks,
    vocals: UploadFile,
    accompaniment: UploadFile,
    vocal_gain_db: float = Form(default=0),
    accompaniment_gain_db: float = Form(default=0),
    project_id: str = Form(default=""),
):
    if project_id and not store.get_project(project_id):
        raise HTTPException(400, "project_id must reference an existing project")
    jid = uuid.uuid4().hex
    vocals_path = INBOX / f"{jid}_vocals_{Path(vocals.filename or 'vocals.wav').name}"
    accompaniment_path = INBOX / f"{jid}_accompaniment_{Path(accompaniment.filename or 'accompaniment.wav').name}"
    try:
        await save_upload_limited(vocals, vocals_path, MAX_AUDIO_UPLOAD_BYTES, "audio")
        await save_upload_limited(accompaniment, accompaniment_path, MAX_AUDIO_UPLOAD_BYTES, "audio")
    except HTTPException:
        vocals_path.unlink(missing_ok=True)
        accompaniment_path.unlink(missing_ok=True)
        raise
    try:
        vocals_metadata = inspect_audio(vocals_path)
        accompaniment_metadata = inspect_audio(accompaniment_path)
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        vocals_path.unlink(missing_ok=True)
        accompaniment_path.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    jobs[jid] = {
        "id": jid,
        "type": "audio_mix",
        "status": "queued",
        "input": [vocals.filename, accompaniment.filename],
        "project_id": project_id,
        "metadata": {"vocals": vocals_metadata, "accompaniment": accompaniment_metadata},
        "retry": {
            "vocals_path": str(vocals_path),
            "accompaniment_path": str(accompaniment_path),
            "vocal_gain_db": vocal_gain_db,
            "accompaniment_gain_db": accompaniment_gain_db,
        },
        "log": [],
    }
    store.save_job(jobs[jid])
    bg.add_task(
        run_mix_job,
        jid,
        vocals_path,
        accompaniment_path,
        vocal_gain_db,
        accompaniment_gain_db,
    )
    return jobs[jid]


async def run_mix_job(jid, vocals_path, accompaniment_path, vocal_gain_db, accompaniment_gain_db):
    job = jobs[jid]
    job["status"] = "running"
    job["log"].append("mixing vocal and accompaniment WAV files")
    store.save_job(job)
    try:
        if cancel_flags.get(jid):
            job.update(status="cancelled")
            store.save_job(job)
            return
        output = OUT / f"mix-{jid}.wav"
        result = await asyncio.to_thread(
            mix_audio,
            vocals_path,
            accompaniment_path,
            output,
            vocal_gain_db,
            accompaniment_gain_db,
        )
        if cancel_flags.get(jid):
            output.unlink(missing_ok=True)
            job.update(status="cancelled")
            store.save_job(job)
            return
        job.update(status="completed", output=f"/api/outputs/{output.name}", result=result)
        job["log"].append("aligned, mixed and exported WAV")
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        job.update(status="failed", error=str(error))
        job["log"].append(str(error))
    store.save_job(job)
@app.post("/api/workflow/cover")
async def cover_workflow(
    bg: BackgroundTasks,
    song: UploadFile,
    model_id: str = Form(),
    index_model_id: str = Form(default=""),
    pitch: int = Form(default=0),
    f0_method: str = Form(default="rmvpe"),
    vocal_gain_db: float = Form(default=0),
    accompaniment_gain_db: float = Form(default=0),
    chunk_seconds: float = Form(default=0),
    overlap_seconds: float = Form(default=1),
    project_id: str = Form(default=""),
):
    if project_id and not store.get_project(project_id):
        raise HTTPException(400, "project_id must reference an existing project")
    demucs_executable = os.environ.get("DEMUCS_EXECUTABLE", "").strip()
    applio_root = os.environ.get("APPLIO_ROOT", "").strip()
    applio_python = os.environ.get("APPLIO_PYTHON", "").strip()
    if not demucs_executable or not applio_root or not applio_python:
        raise HTTPException(503, "Demucs and Applio engine environments must both be configured")
    model = store.get_model(model_id)
    if not model or model["type"] != "rvc" or Path(model["name"]).suffix.lower() != ".pth":
        raise HTTPException(400, "a registered RVC .pth inference model is required")
    index_path = ""
    if index_model_id:
        index_model = store.get_model(index_model_id)
        if not index_model or index_model["type"] != "rvc" or Path(index_model["name"]).suffix.lower() != ".index":
            raise HTTPException(400, "index_model_id must reference a registered RVC .index file")
        index_path = index_model["path"]
    jid = uuid.uuid4().hex
    safe = Path(song.filename or "song.wav").name
    source = INBOX / f"{jid}_{safe}"
    await save_upload_limited(song, source, MAX_AUDIO_UPLOAD_BYTES, "audio")
    try:
        metadata = inspect_audio(source)
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        source.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    conversion_parameters = {
        "model_path": model["path"],
        "index_path": index_path,
        "pitch": pitch,
        "f0_method": f0_method,
    }
    jobs[jid] = {
        "id": jid,
        "type": "cover_workflow",
        "status": "queued",
        "input": safe,
        "project_id": project_id,
        "model_id": model_id,
        "metadata": metadata,
        "retry": {
            "source": str(source),
            "conversion_parameters": conversion_parameters,
            "vocal_gain_db": vocal_gain_db,
            "accompaniment_gain_db": accompaniment_gain_db,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
        },
        "log": [],
    }
    store.save_job(jobs[jid])
    bg.add_task(
        run_cover_job,
        jid,
        source,
        demucs_executable,
        Path(applio_root),
        applio_python,
        conversion_parameters,
        vocal_gain_db,
        accompaniment_gain_db,
        chunk_seconds,
        overlap_seconds,
    )
    return jobs[jid]


async def run_cover_job(
    jid,
    source,
    demucs_executable,
    applio_root,
    applio_python,
    conversion_parameters,
    vocal_gain_db,
    accompaniment_gain_db,
    chunk_seconds,
    overlap_seconds,
):
    job = jobs[jid]
    job["status"] = "running"
    job["log"].append("running separation, RVC conversion, quality comparison and mix")
    store.save_job(job)

    def report_progress(stage, value):
        job.update(progress_stage=stage, progress=value)
        store.save_job(job)

    try:
        if cancel_flags.get(jid):
            job.update(status="cancelled")
            store.save_job(job)
            return
        output = OUT / f"cover-{jid}.wav"
        result = await run_cover_workflow(
            DemucsAdapter(demucs_executable, OUT),
            ApplioAdapter(applio_root, applio_python, OUT),
            source,
            output,
            conversion_parameters,
            vocal_gain_db=vocal_gain_db,
            accompaniment_gain_db=accompaniment_gain_db,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            workspace=OUT / f"cover-chunks-{jid}",
            progress_callback=report_progress,
        )
        if cancel_flags.get(jid):
            output.unlink(missing_ok=True)
            job.update(status="cancelled")
            store.save_job(job)
            return
        result.pop("artifacts", None)
        job.update(status="completed", output=f"/api/outputs/{output.name}", result=result)
        job["log"].append("completed preset cover workflow; subjective listening still required")
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        job.update(status="failed", error=str(error))
        job["log"].append(str(error))
    store.save_job(job)


@app.post("/api/workflow/separate")
async def separation_workflow(
    bg: BackgroundTasks,
    input_audio: UploadFile,
    model: str = Form(default="htdemucs"),
    device: str = Form(default=""),
    project_id: str = Form(default=""),
):
    if project_id and not store.get_project(project_id):
        raise HTTPException(400, "project_id must reference an existing project")
    executable = os.environ.get("DEMUCS_EXECUTABLE", "").strip()
    if not executable:
        raise HTTPException(503, "Demucs engine executable is not configured")
    jid = uuid.uuid4().hex
    safe = Path(input_audio.filename or "song.wav").name
    source = INBOX / f"{jid}_{safe}"
    await save_upload_limited(input_audio, source, MAX_AUDIO_UPLOAD_BYTES, "audio")
    try:
        metadata = inspect_audio(source)
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        source.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    jobs[jid] = {
        "id": jid,
        "type": "vocal_separation",
        "status": "queued",
        "input": safe,
        "project_id": project_id,
        "metadata": metadata,
        "retry": {
            "input_audio": str(source),
            "parameters": {"model": model, "device": device},
        },
        "log": [],
    }
    store.save_job(jobs[jid])
    request = EngineRequest(input_audio=str(source), parameters={"model": model, "device": device})
    bg.add_task(run_separation_job, jid, executable, request)
    return jobs[jid]


async def run_separation_job(jid, executable, request):
    job = jobs[jid]
    job["status"] = "running"
    job["log"].append("submitted to Demucs CLI")
    store.save_job(job)
    try:
        if cancel_flags.get(jid):
            job.update(status="cancelled")
            store.save_job(job)
            return
        result = await DemucsAdapter(executable, OUT).run(request)
        if cancel_flags.get(jid):
            for path in result.artifacts.values():
                Path(path).unlink(missing_ok=True)
            job.update(status="cancelled")
            store.save_job(job)
            return
        artifacts = {name: f"/api/outputs/{Path(path).name}" for name, path in result.artifacts.items()}
        job.update(
            status="completed",
            output=artifacts["vocals"],
            artifacts=artifacts,
            result={"sample_rate": result.sample_rate, "engine": result.engine, **result.metadata},
        )
        job["log"].append("received and validated vocal/accompaniment WAV stems")
    except (RuntimeError, ValueError, OSError) as error:
        job.update(status="failed", error=str(error))
        job["log"].append(str(error))
    store.save_job(job)


@app.post("/api/workflow/singing")
async def singing_workflow(
    bg: BackgroundTasks,
    ds_file: UploadFile,
    experiment: str = Form(),
    speaker: str = Form(default=""),
    language: str = Form(default=""),
    key: int = Form(default=0),
    steps: int | None = Form(default=None),
    project_id: str = Form(default=""),
):
    if project_id and not store.get_project(project_id):
        raise HTTPException(400, "project_id must reference an existing project")
    diffsinger_root = os.environ.get("DIFFSINGER_ROOT", "").strip()
    diffsinger_python = os.environ.get("DIFFSINGER_PYTHON", "").strip()
    if not diffsinger_root or not diffsinger_python:
        raise HTTPException(503, "DiffSinger engine environment is not configured")
    if not experiment.strip():
        raise HTTPException(400, "DiffSinger experiment name is required")
    safe = Path(ds_file.filename or "song.ds").name
    if Path(safe).suffix.lower() != ".ds":
        raise HTTPException(400, "singing input must be a DiffSinger .ds file")
    jid = uuid.uuid4().hex
    source = INBOX / f"{jid}_{safe}"
    await save_upload_limited(ds_file, source, MAX_DOCUMENT_UPLOAD_BYTES, "DiffSinger DS")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        source.unlink(missing_ok=True)
        raise HTTPException(400, "DiffSinger DS file must contain valid UTF-8 JSON") from error
    if not isinstance(document, (dict, list)) or not document:
        source.unlink(missing_ok=True)
        raise HTTPException(400, "DiffSinger DS file must contain at least one segment")
    jobs[jid] = {
        "id": jid,
        "type": "singing_synthesis",
        "status": "queued",
        "input": safe,
        "project_id": project_id,
        "metadata": {"format": "DiffSinger DS", "experiment": experiment.strip()},
        "retry": {
            "parameters": {
                "ds_path": str(source),
                "experiment": experiment.strip(),
                "speaker": speaker.strip(),
                "language": language.strip(),
                "key": key,
                "steps": steps,
            }
        },
        "log": [],
    }
    store.save_job(jobs[jid])
    request = EngineRequest(
        parameters={
            "ds_path": str(source),
            "experiment": experiment.strip(),
            "speaker": speaker.strip(),
            "language": language.strip(),
            "key": key,
            "steps": steps,
        }
    )
    bg.add_task(run_singing_job, jid, Path(diffsinger_root), diffsinger_python, request)
    return jobs[jid]


async def run_singing_job(jid, diffsinger_root, diffsinger_python, request):
    job = jobs[jid]
    job["status"] = "running"
    job["log"].append("submitted DiffSinger DS file to acoustic inference")
    store.save_job(job)
    try:
        if cancel_flags.get(jid):
            job.update(status="cancelled")
            store.save_job(job)
            return
        result = await DiffSingerAdapter(diffsinger_root, diffsinger_python, OUT).run(request)
        output = Path(result.output_audio)
        if cancel_flags.get(jid):
            output.unlink(missing_ok=True)
            job.update(status="cancelled")
            store.save_job(job)
            return
        job.update(
            status="completed",
            output=f"/api/outputs/{output.name}",
            result={"sample_rate": result.sample_rate, "engine": result.engine, **result.metadata},
        )
        job["log"].append("received and validated DiffSinger WAV output")
    except (RuntimeError, ValueError, OSError) as error:
        job.update(status="failed", error=str(error))
        job["log"].append(str(error))
    store.save_job(job)


@app.post("/api/workflow/train-rvc")
async def train_rvc_workflow(
    bg: BackgroundTasks,
    dataset_id: str = Form(),
    model_name: str = Form(),
    total_epoch: int = Form(default=200),
    batch_size: int = Form(default=4),
    sample_rate: int = Form(default=40000),
    gpu: str = Form(default="0"),
    project_id: str = Form(default=""),
):
    if project_id and not store.get_project(project_id):
        raise HTTPException(400, "project_id must reference an existing project")
    applio_root = os.environ.get("APPLIO_ROOT", "").strip()
    applio_python = os.environ.get("APPLIO_PYTHON", "").strip()
    if not applio_root or not applio_python:
        raise HTTPException(503, "Applio engine environment is not configured")
    dataset = store.get_dataset(dataset_id)
    if not dataset or not Path(dataset["path"]).is_dir():
        raise HTTPException(400, "a registered training dataset is required")
    if not 1 <= total_epoch <= 10000 or not 1 <= batch_size <= 1024:
        raise HTTPException(400, "training epoch or batch size is outside the supported range")
    jid = uuid.uuid4().hex
    request = TrainingRequest(
        model_name=model_name,
        dataset_path=dataset["path"],
        parameters={
            "total_epoch": total_epoch,
            "batch_size": batch_size,
            "sample_rate": sample_rate,
            "gpu": gpu,
        },
    )
    jobs[jid] = {
        "id": jid,
        "type": "rvc_training",
        "status": "queued",
        "input": dataset_id,
        "project_id": project_id,
        "metadata": {"model_name": model_name, "dataset": dataset["name"], "total_epoch": total_epoch},
        "retry": {"model_name": model_name, "parameters": request.parameters},
        "log": [],
    }
    store.save_job(jobs[jid])
    bg.add_task(run_training_job, jid, Path(applio_root), applio_python, request, dataset)
    return jobs[jid]


async def run_training_job(jid, applio_root, applio_python, request, dataset):
    job = jobs[jid]
    job["status"] = "running"
    job["log"].append("submitted Applio preprocess/extract/train/index pipeline")
    store.save_job(job)
    try:
        if cancel_flags.get(jid):
            job.update(status="cancelled")
            store.save_job(job)
            return
        result = await ApplioTrainingAdapter(applio_root, applio_python).run(request)
        if cancel_flags.get(jid):
            job.update(status="cancelled")
            store.save_job(job)
            return
        model_ids = []
        for artifact_type, source_name in result.artifacts.items():
            source = Path(source_name)
            model_id = uuid.uuid4().hex
            target_dir = MODELS / model_id
            target_dir.mkdir(parents=True, exist_ok=False)
            target = target_dir / source.name
            shutil.copyfile(source, target)
            model = {
                "id": model_id,
                "name": target.name,
                "type": "rvc",
                "size": target.stat().st_size,
                "sha256": sha256_file(target),
                "license": dataset["license"],
                "source_url": "",
                "status": "trained_unverified",
                "path": str(target),
                "training_job_id": jid,
                "artifact_type": artifact_type,
            }
            store.save_model(model)
            model_ids.append(model_id)
        job.update(
            status="completed",
            artifacts={"model_ids": model_ids},
            result={"engine": result.engine, **result.metadata},
        )
        job["log"].append("registered unverified RVC training artifacts")
    except (RuntimeError, ValueError, OSError) as error:
        job.update(status="failed", error=str(error))
        job["log"].append(str(error))
    store.save_job(job)


@app.post("/api/workflow/voice-conversion")
async def voice_conversion_workflow(
    bg: BackgroundTasks,
    input_audio: UploadFile,
    model_id: str = Form(),
    index_model_id: str = Form(default=""),
    pitch: int = Form(default=0),
    index_rate: float = Form(default=0.5),
    protect: float = Form(default=0.33),
    f0_method: str = Form(default="rmvpe"),
    project_id: str = Form(default=""),
):
    if project_id and not store.get_project(project_id):
        raise HTTPException(400, "project_id must reference an existing project")
    applio_root = os.environ.get("APPLIO_ROOT", "").strip()
    applio_python = os.environ.get("APPLIO_PYTHON", "").strip()
    if not applio_root or not applio_python:
        raise HTTPException(503, "Applio engine environment is not configured")
    model = store.get_model(model_id)
    if not model or model["type"] != "rvc" or Path(model["name"]).suffix.lower() != ".pth":
        raise HTTPException(400, "a registered RVC .pth inference model is required")
    index_path = ""
    if index_model_id:
        index_model = store.get_model(index_model_id)
        if not index_model or index_model["type"] != "rvc" or Path(index_model["name"]).suffix.lower() != ".index":
            raise HTTPException(400, "index_model_id must reference a registered RVC .index file")
        index_path = index_model["path"]
    jid = uuid.uuid4().hex
    safe = Path(input_audio.filename or "input.wav").name
    source = INBOX / f"{jid}_{safe}"
    await save_upload_limited(input_audio, source, MAX_AUDIO_UPLOAD_BYTES, "audio")
    try:
        metadata = inspect_audio(source)
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        source.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    jobs[jid] = {
        "id": jid,
        "type": "voice_conversion",
        "status": "queued",
        "input": safe,
        "project_id": project_id,
        "model_id": model_id,
        "metadata": metadata,
        "retry": {
            "input_audio": str(source),
            "model_id": model_id,
            "parameters": {
                "model_path": model["path"],
                "index_path": index_path,
                "pitch": pitch,
                "index_rate": index_rate,
                "protect": protect,
                "f0_method": f0_method,
            },
        },
        "log": [],
    }
    store.save_job(jobs[jid])
    request = EngineRequest(
        input_audio=str(source),
        model_id=model_id,
        parameters={
            "model_path": model["path"],
            "index_path": index_path,
            "pitch": pitch,
            "index_rate": index_rate,
            "protect": protect,
            "f0_method": f0_method,
        },
    )
    bg.add_task(run_voice_conversion_job, jid, Path(applio_root), applio_python, request)
    return jobs[jid]


async def run_voice_conversion_job(jid, applio_root, applio_python, request):
    job = jobs[jid]
    job["status"] = "running"
    job["log"].append("submitted to Applio CLI")
    store.save_job(job)
    try:
        if cancel_flags.get(jid):
            job.update(status="cancelled")
            store.save_job(job)
            return
        result = await ApplioAdapter(applio_root, applio_python, OUT).run(request)
        output = Path(result.output_audio)
        if cancel_flags.get(jid):
            output.unlink(missing_ok=True)
            job.update(status="cancelled")
            store.save_job(job)
            return
        job.update(
            status="completed",
            output=f"/api/outputs/{output.name}",
            result={"sample_rate": result.sample_rate, "engine": result.engine, **result.metadata},
        )
        job["log"].append("received and validated RVC WAV output")
    except (RuntimeError, ValueError, OSError) as error:
        job.update(status="failed", error=str(error))
        job["log"].append(str(error))
    store.save_job(job)
@app.post("/api/workflow/tts")
async def tts_workflow(
    bg: BackgroundTasks,
    reference_audio: UploadFile,
    text: str = Form(),
    text_lang: str = Form(default="zh"),
    prompt_lang: str = Form(default="zh"),
    prompt_text: str = Form(default=""),
    project_id: str = Form(default=""),
):
    if project_id and not store.get_project(project_id):
        raise HTTPException(400, "project_id must reference an existing project")
    engine_url = os.environ.get("GPT_SOVITS_URL", "").strip()
    if not engine_url:
        raise HTTPException(503, "GPT-SoVITS engine URL is not configured")
    if not text.strip():
        raise HTTPException(400, "text is required")
    jid = uuid.uuid4().hex
    safe = Path(reference_audio.filename or "reference.wav").name
    source = INBOX / f"{jid}_{safe}"
    await save_upload_limited(reference_audio, source, MAX_AUDIO_UPLOAD_BYTES, "audio")
    try:
        metadata = inspect_audio(source)
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as error:
        source.unlink(missing_ok=True)
        raise HTTPException(400, str(error)) from error
    jobs[jid] = {
        "id": jid,
        "type": "tts",
        "status": "queued",
        "input": safe,
        "project_id": project_id,
        "metadata": metadata,
        "retry": {
            "reference_audio": str(source),
            "text": text.strip(),
            "parameters": {
                "text_lang": text_lang,
                "prompt_lang": prompt_lang,
                "prompt_text": prompt_text,
            },
        },
        "log": [],
    }
    store.save_job(jobs[jid])
    request = EngineRequest(
        text=text.strip(),
        reference_audio=str(source),
        parameters={"text_lang": text_lang, "prompt_lang": prompt_lang, "prompt_text": prompt_text},
    )
    bg.add_task(run_tts_job, jid, engine_url, request)
    return jobs[jid]


async def run_tts_job(jid, engine_url, request):
    job = jobs[jid]
    job["status"] = "running"
    job["log"].append("submitted to GPT-SoVITS API v2")
    store.save_job(job)
    try:
        if cancel_flags.get(jid):
            job.update(status="cancelled")
            store.save_job(job)
            return
        result = await GptSovitsAdapter(engine_url, OUT).run(request)
        output = Path(result.output_audio)
        if cancel_flags.get(jid):
            output.unlink(missing_ok=True)
            job.update(status="cancelled")
            store.save_job(job)
            return
        job.update(
            status="completed",
            output=f"/api/outputs/{output.name}",
            result={"sample_rate": result.sample_rate, "engine": result.engine, **result.metadata},
        )
        job["log"].append("received and validated WAV output")
    except (RuntimeError, ValueError, OSError, httpx.HTTPError) as error:
        job.update(status="failed", error=str(error))
        job["log"].append(str(error))
    store.save_job(job)
async def run_job(jid,src):
    job=jobs[jid]; job["status"]="running"; job["log"].append("validated input")
    store.save_job(job)
    try:
        if cancel_flags.get(jid):
            job.update(status="cancelled"); store.save_job(job); return
        dst=OUT/f"{jid}.wav"; result=await asyncio.to_thread(process_audio,src,dst)
        if cancel_flags.get(jid):
            dst.unlink(missing_ok=True); job.update(status="cancelled"); store.save_job(job); return
        job.update(status="completed",output=f"/api/outputs/{jid}.wav",result=result); job["log"].append("normalized and exported WAV")
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as e: job.update(status="failed",error=str(e)); job["log"].append(str(e))
    store.save_job(job)
@app.get("/api/jobs/{jid}")
async def get_job(jid):
    job=jobs.get(jid) or store.get_job(jid)
    if not job: raise HTTPException(404,"job not found")
    return job
@app.post("/api/jobs/{jid}/cancel")
async def cancel_job(jid):
    job=jobs.get(jid) or store.get_job(jid)
    if not job: raise HTTPException(404,"job not found")
    if job["status"] not in ("queued", "running"):
        raise HTTPException(409, "job is already finished")
    cancel_flags[jid]=True
    job["log"].append("cancellation requested")
    store.save_job(job)
    return {"id":jid,"status":"cancellation_requested"}
@app.post("/api/jobs/{jid}/retry")
async def retry_job(jid, bg: BackgroundTasks):
    job=jobs.get(jid) or store.get_job(jid)
    if not job: raise HTTPException(404,"job not found")
    if job["status"] in ("queued", "running"):
        raise HTTPException(409,"active job cannot be retried")
    if job.get("type") == "rvc_training":
        applio_root = os.environ.get("APPLIO_ROOT", "").strip()
        applio_python = os.environ.get("APPLIO_PYTHON", "").strip()
        if not applio_root or not applio_python:
            raise HTTPException(503, "Applio engine environment is not configured")
        dataset = store.get_dataset(job.get("input", ""))
        if not dataset or not Path(dataset["path"]).is_dir():
            raise HTTPException(404, "registered training dataset is missing")
        retry = job.get("retry") or {}
        new_id = uuid.uuid4().hex
        jobs[new_id] = {
            "id": new_id,
            "type": "rvc_training",
            "status": "queued",
            "input": job["input"],
            "project_id": job.get("project_id", ""),
            "metadata": job["metadata"],
            "retry": retry,
            "log": [f"retried from {jid}"],
        }
        cancel_flags.pop(new_id, None)
        store.save_job(jobs[new_id])
        request = TrainingRequest(
            model_name=retry.get("model_name", job["metadata"].get("model_name", "")),
            dataset_path=dataset["path"],
            parameters=retry.get("parameters", {}),
        )
        bg.add_task(run_training_job, new_id, Path(applio_root), applio_python, request, dataset)
        return jobs[new_id]
    if job.get("type") == "audio_mix":
        retry = job.get("retry") or {}
        vocals_path = Path(retry.get("vocals_path", ""))
        accompaniment_path = Path(retry.get("accompaniment_path", ""))
        if not vocals_path.is_file() or not accompaniment_path.is_file():
            raise HTTPException(404, "original vocal or accompaniment audio is missing")
        new_id = uuid.uuid4().hex
        jobs[new_id] = {
            "id": new_id,
            "type": "audio_mix",
            "status": "queued",
            "input": job["input"],
            "project_id": job.get("project_id", ""),
            "metadata": job["metadata"],
            "retry": retry,
            "log": [f"retried from {jid}"],
        }
        cancel_flags.pop(new_id, None)
        store.save_job(jobs[new_id])
        bg.add_task(
            run_mix_job,
            new_id,
            vocals_path,
            accompaniment_path,
            retry.get("vocal_gain_db", 0),
            retry.get("accompaniment_gain_db", 0),
        )
        return jobs[new_id]
    if job.get("type") == "cover_workflow":
        demucs_executable = os.environ.get("DEMUCS_EXECUTABLE", "").strip()
        applio_root = os.environ.get("APPLIO_ROOT", "").strip()
        applio_python = os.environ.get("APPLIO_PYTHON", "").strip()
        if not demucs_executable or not applio_root or not applio_python:
            raise HTTPException(503, "Demucs and Applio engine environments must both be configured")
        retry = job.get("retry") or {}
        source = Path(retry.get("source", ""))
        parameters = retry.get("conversion_parameters") or {}
        if not source.is_file():
            raise HTTPException(404, "original song audio is missing")
        if not Path(parameters.get("model_path", "")).is_file():
            raise HTTPException(404, "registered RVC model file is missing")
        new_id = uuid.uuid4().hex
        jobs[new_id] = {
            "id": new_id,
            "type": "cover_workflow",
            "status": "queued",
            "input": job["input"],
            "project_id": job.get("project_id", ""),
            "model_id": job.get("model_id", ""),
            "metadata": job["metadata"],
            "retry": retry,
            "log": [f"retried from {jid}"],
        }
        cancel_flags.pop(new_id, None)
        store.save_job(jobs[new_id])
        bg.add_task(
            run_cover_job,
            new_id,
            source,
            demucs_executable,
            Path(applio_root),
            applio_python,
            parameters,
            retry.get("vocal_gain_db", 0),
            retry.get("accompaniment_gain_db", 0),
            retry.get("chunk_seconds", 0),
            retry.get("overlap_seconds", 1),
        )
        return jobs[new_id]
    if job.get("type") == "singing_synthesis":
        diffsinger_root = os.environ.get("DIFFSINGER_ROOT", "").strip()
        diffsinger_python = os.environ.get("DIFFSINGER_PYTHON", "").strip()
        if not diffsinger_root or not diffsinger_python:
            raise HTTPException(503, "DiffSinger engine environment is not configured")
        retry = job.get("retry") or {}
        parameters = retry.get("parameters") or {}
        if not Path(parameters.get("ds_path", "")).is_file():
            raise HTTPException(404, "original DiffSinger DS file is missing")
        new_id = uuid.uuid4().hex
        jobs[new_id] = {
            "id": new_id,
            "type": "singing_synthesis",
            "status": "queued",
            "input": job["input"],
            "project_id": job.get("project_id", ""),
            "metadata": job["metadata"],
            "retry": retry,
            "log": [f"retried from {jid}"],
        }
        cancel_flags.pop(new_id, None)
        store.save_job(jobs[new_id])
        request = EngineRequest(parameters=parameters)
        bg.add_task(run_singing_job, new_id, Path(diffsinger_root), diffsinger_python, request)
        return jobs[new_id]
    if job.get("type") == "vocal_separation":
        executable = os.environ.get("DEMUCS_EXECUTABLE", "").strip()
        if not executable:
            raise HTTPException(503, "Demucs engine executable is not configured")
        retry = job.get("retry") or {}
        input_audio = Path(retry.get("input_audio", ""))
        if not input_audio.is_file():
            raise HTTPException(404, "original input audio is missing")
        new_id = uuid.uuid4().hex
        jobs[new_id] = {
            "id": new_id,
            "type": "vocal_separation",
            "status": "queued",
            "input": job["input"],
            "project_id": job.get("project_id", ""),
            "metadata": job["metadata"],
            "retry": retry,
            "log": [f"retried from {jid}"],
        }
        cancel_flags.pop(new_id, None)
        store.save_job(jobs[new_id])
        request = EngineRequest(input_audio=str(input_audio), parameters=retry.get("parameters", {}))
        bg.add_task(run_separation_job, new_id, executable, request)
        return jobs[new_id]
    if job.get("type") == "voice_conversion":
        applio_root = os.environ.get("APPLIO_ROOT", "").strip()
        applio_python = os.environ.get("APPLIO_PYTHON", "").strip()
        if not applio_root or not applio_python:
            raise HTTPException(503, "Applio engine environment is not configured")
        retry = job.get("retry") or {}
        input_audio = Path(retry.get("input_audio", ""))
        parameters = retry.get("parameters") or {}
        if not input_audio.is_file():
            raise HTTPException(404, "original input audio is missing")
        if not Path(parameters.get("model_path", "")).is_file():
            raise HTTPException(404, "registered RVC model file is missing")
        new_id = uuid.uuid4().hex
        jobs[new_id] = {
            "id": new_id,
            "type": "voice_conversion",
            "status": "queued",
            "input": job["input"],
            "project_id": job.get("project_id", ""),
            "model_id": retry.get("model_id", job.get("model_id", "")),
            "metadata": job["metadata"],
            "retry": retry,
            "log": [f"retried from {jid}"],
        }
        cancel_flags.pop(new_id, None)
        store.save_job(jobs[new_id])
        request = EngineRequest(
            input_audio=str(input_audio),
            model_id=jobs[new_id]["model_id"],
            parameters=parameters,
        )
        bg.add_task(run_voice_conversion_job, new_id, Path(applio_root), applio_python, request)
        return jobs[new_id]
    if job.get("type") == "tts":
        engine_url = os.environ.get("GPT_SOVITS_URL", "").strip()
        if not engine_url:
            raise HTTPException(503, "GPT-SoVITS engine URL is not configured")
        retry = job.get("retry") or {}
        reference_audio = Path(retry.get("reference_audio", ""))
        if not reference_audio.is_file():
            raise HTTPException(404, "original reference audio is missing")
        new_id = uuid.uuid4().hex
        jobs[new_id] = {
            "id": new_id,
            "type": "tts",
            "status": "queued",
            "input": job["input"],
            "project_id": job.get("project_id", ""),
            "metadata": job["metadata"],
            "retry": retry,
            "log": [f"retried from {jid}"],
        }
        cancel_flags.pop(new_id, None)
        store.save_job(jobs[new_id])
        request = EngineRequest(
            text=retry.get("text", ""),
            reference_audio=str(reference_audio),
            parameters=retry.get("parameters", {}),
        )
        bg.add_task(run_tts_job, new_id, engine_url, request)
        return jobs[new_id]
    if job.get("type", "audio_processing") != "audio_processing":
        raise HTTPException(409, "retry is not implemented for this job type")
    old=INBOX/f"{jid}_{Path(job['input']).name}"
    if not old.is_file(): raise HTTPException(404,"original input file is missing")
    new_id=uuid.uuid4().hex
    retry_input=INBOX/f"{new_id}_{Path(job['input']).name}"
    retry_input.write_bytes(old.read_bytes())
    jobs[new_id]={"id":new_id,"type":"audio_processing","status":"queued","input":job["input"],"metadata":job["metadata"],"log":[f"retried from {jid}"]}
    cancel_flags.pop(new_id,None)
    store.save_job(jobs[new_id])
    bg.add_task(run_job,new_id,retry_input)
    return jobs[new_id]
@app.get("/api/outputs/{name}")
async def output(name):
    p=OUT/Path(name).name
    if not p.is_file(): raise HTTPException(404,"output not found")
    return FileResponse(p, media_type="audio/wav", filename=p.name)
