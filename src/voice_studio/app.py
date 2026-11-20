import asyncio
import os
import uuid
from pathlib import Path

import httpx
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .adapters.gpt_sovits import GptSovitsAdapter
from .audio import inspect_audio, process_audio
from .engines import EngineRequest
from .models import sha256_file, validate_model_filename
from .store import Store
from .ui import PAGE

ROOT = Path(os.environ.get("AI_VOICE_STUDIO_DATA_DIR", Path.cwd() / "work" / "runtime")).resolve()
INBOX, OUT, MODELS = ROOT / "audio", ROOT / "outputs", ROOT / "models"
for p in (INBOX, OUT, MODELS): p.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="AI Voice Studio", version="0.1.0")
jobs = {}
cancel_flags = {}
store = Store(ROOT / "studio.sqlite3")
store.recover_incomplete_jobs()
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
@app.get("/api/jobs")
async def list_jobs(): return store.list_jobs()
@app.get("/api/capabilities")
async def capabilities():
    tts_status = "configured_unverified" if os.environ.get("GPT_SOVITS_URL") else "interface_only_dependencies_missing"
    return {
        "audio_processing": "verified_real_processing",
        "tts": tts_status,
        "voice_conversion": "interface_only_dependencies_missing",
        "vocal_separation": "interface_only_dependencies_missing",
        "singing_synthesis": "interface_only_dependencies_missing",
        "model_training": "not_developed",
    }
@app.get("/api/engines")
async def engines():
    return [{"id": "gpt-sovits", "capability": "tts", "configured": bool(os.environ.get("GPT_SOVITS_URL")), "verification": "not_run"}]
@app.get("/api/models")
async def list_models(): return store.list_models()
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
    size = 0
    with target.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            handle.write(chunk)
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
    }
    store.save_model(model)
    return model
@app.post("/api/workflow/audio")
async def workflow(file: UploadFile, bg: BackgroundTasks):
    jid=uuid.uuid4().hex
    safe=Path(file.filename or "audio.wav").name
    src=INBOX/f"{jid}_{safe}"
    src.write_bytes(await file.read())
    try: meta=inspect_audio(src)
    except (RuntimeError, ValueError, OSError, sf.LibsndfileError) as e: src.unlink(missing_ok=True); raise HTTPException(400,str(e)) from e
    jobs[jid]={"id":jid,"type":"audio_processing","status":"queued","input":safe,"metadata":meta,"log":[]}
    store.save_job(jobs[jid])
    bg.add_task(run_job,jid,src)
    return jobs[jid]
@app.post("/api/workflow/tts")
async def tts_workflow(
    bg: BackgroundTasks,
    reference_audio: UploadFile,
    text: str = Form(),
    text_lang: str = Form(default="zh"),
    prompt_lang: str = Form(default="zh"),
    prompt_text: str = Form(default=""),
):
    engine_url = os.environ.get("GPT_SOVITS_URL", "").strip()
    if not engine_url:
        raise HTTPException(503, "GPT-SoVITS engine URL is not configured")
    if not text.strip():
        raise HTTPException(400, "text is required")
    jid = uuid.uuid4().hex
    safe = Path(reference_audio.filename or "reference.wav").name
    source = INBOX / f"{jid}_{safe}"
    source.write_bytes(await reference_audio.read())
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
        "metadata": metadata,
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
