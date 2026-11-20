import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import soundfile as sf

from ..engines import EngineRequest, EngineResult

ProcessExecutor = Callable[[Sequence[str], Path, float], Awaitable[tuple[int, str, str]]]


async def execute_process(command: Sequence[str], cwd: Path, timeout: float) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"Applio inference exceeded {timeout:g} seconds") from None
    return process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


class ApplioAdapter:
    name = "applio-cli"
    capability = "voice_conversion"

    def __init__(
        self,
        applio_root: Path,
        python_executable: str,
        output_dir: Path,
        timeout_seconds: float = 900,
        executor: ProcessExecutor = execute_process,
    ):
        self.applio_root = applio_root
        self.python_executable = python_executable
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.executor = executor

    def build_command(self, request: EngineRequest, output: Path) -> list[str]:
        params = request.parameters
        if not request.input_audio:
            raise ValueError("Applio requires input audio")
        model_path = str(params.get("model_path", ""))
        if not model_path:
            raise ValueError("Applio requires an RVC inference model path")
        command = [
            self.python_executable,
            "core.py",
            "infer",
            "--input-path",
            request.input_audio,
            "--output-path",
            str(output),
            "--pth-path",
            model_path,
            "--index-path",
            str(params.get("index_path", "")),
            "--pitch",
            str(params.get("pitch", 0)),
            "--index-rate",
            str(params.get("index_rate", 0.5)),
            "--volume-envelope",
            str(params.get("volume_envelope", 1.0)),
            "--protect",
            str(params.get("protect", 0.33)),
            "--f0-method",
            str(params.get("f0_method", "rmvpe")),
            "--embedder-model",
            str(params.get("embedder_model", "contentvec")),
            "--export-format",
            "WAV",
        ]
        if params.get("split_audio", True):
            command.append("--split-audio")
        if params.get("f0_autotune", False):
            command.append("--f0-autotune")
        return command

    async def run(self, request: EngineRequest) -> EngineResult:
        if not (self.applio_root / "core.py").is_file():
            raise RuntimeError("Applio core.py was not found in the configured engine directory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"rvc-{__import__('uuid').uuid4().hex}.wav"
        command = self.build_command(request, output)
        code, stdout, stderr = await self.executor(command, self.applio_root, self.timeout_seconds)
        if code != 0:
            raise RuntimeError(f"Applio inference failed ({code}): {(stderr or stdout).strip()}")
        if not output.is_file():
            raise RuntimeError("Applio reported success but did not create the output WAV")
        try:
            info = sf.info(output)
        except (RuntimeError, sf.LibsndfileError) as error:
            output.unlink(missing_ok=True)
            raise RuntimeError("Applio created an invalid WAV output") from error
        return EngineResult(
            output_audio=str(output),
            sample_rate=info.samplerate,
            engine=self.name,
            metadata={"duration": info.duration, "stdout": stdout[-2000:]},
        )
