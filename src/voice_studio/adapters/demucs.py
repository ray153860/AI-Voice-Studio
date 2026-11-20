import shutil
import uuid
from pathlib import Path

import soundfile as sf

from ..engines import EngineRequest, EngineResult
from .applio import ProcessExecutor, execute_process


class DemucsAdapter:
    name = "demucs-cli"
    capability = "vocal_separation"

    def __init__(
        self,
        executable: str,
        output_dir: Path,
        timeout_seconds: float = 1800,
        executor: ProcessExecutor = execute_process,
    ):
        self.executable = executable
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.executor = executor

    async def run(self, request: EngineRequest) -> EngineResult:
        if not request.input_audio:
            raise ValueError("Demucs requires input audio")
        input_path = Path(request.input_audio)
        if not input_path.is_file():
            raise ValueError("Demucs input audio does not exist")
        model = str(request.parameters.get("model", "htdemucs"))
        work_dir = self.output_dir / f"demucs-{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.executable,
            "--two-stems=vocals",
            "-n",
            model,
            "-o",
            str(work_dir),
            str(input_path),
        ]
        device = request.parameters.get("device")
        if device:
            command[1:1] = ["-d", str(device)]
        code, stdout, stderr = await self.executor(command, work_dir, self.timeout_seconds)
        if code != 0:
            raise RuntimeError(f"Demucs separation failed ({code}): {(stderr or stdout).strip()}")
        stem_dir = work_dir / model / input_path.stem
        vocals_source = stem_dir / "vocals.wav"
        accompaniment_source = stem_dir / "no_vocals.wav"
        if not vocals_source.is_file() or not accompaniment_source.is_file():
            raise RuntimeError("Demucs did not create both vocals.wav and no_vocals.wav")
        vocals = self.output_dir / f"vocals-{uuid.uuid4().hex}.wav"
        accompaniment = self.output_dir / f"accompaniment-{uuid.uuid4().hex}.wav"
        shutil.copyfile(vocals_source, vocals)
        shutil.copyfile(accompaniment_source, accompaniment)
        try:
            vocals_info = sf.info(vocals)
            accompaniment_info = sf.info(accompaniment)
        except (RuntimeError, sf.LibsndfileError) as error:
            vocals.unlink(missing_ok=True)
            accompaniment.unlink(missing_ok=True)
            raise RuntimeError("Demucs created invalid WAV stems") from error
        return EngineResult(
            output_audio=str(vocals),
            sample_rate=vocals_info.samplerate,
            engine=self.name,
            metadata={
                "model": model,
                "vocals_duration": vocals_info.duration,
                "accompaniment_duration": accompaniment_info.duration,
                "stdout": stdout[-2000:],
            },
            artifacts={"vocals": str(vocals), "accompaniment": str(accompaniment)},
        )
