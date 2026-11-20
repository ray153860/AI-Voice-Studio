import asyncio
import uuid
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
        raise RuntimeError(f"DiffSinger inference exceeded {timeout:g} seconds") from None
    return process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


class DiffSingerAdapter:
    name = "diffsinger-cli"
    capability = "singing_synthesis"

    def __init__(
        self,
        diffsinger_root: Path,
        python_executable: str,
        output_dir: Path,
        timeout_seconds: float = 1800,
        executor: ProcessExecutor = execute_process,
    ):
        self.diffsinger_root = diffsinger_root
        self.python_executable = python_executable
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.executor = executor

    def build_command(self, request: EngineRequest, output_dir: Path, title: str) -> list[str]:
        params = request.parameters
        ds_path = Path(str(params.get("ds_path", "")))
        if not ds_path.is_file() or ds_path.suffix.lower() != ".ds":
            raise ValueError("DiffSinger requires an existing DS file")
        experiment = str(params.get("experiment", "")).strip()
        if not experiment:
            raise ValueError("DiffSinger requires an experiment name")
        command = [
            self.python_executable,
            "scripts/infer.py",
            "acoustic",
            str(ds_path),
            "--exp",
            experiment,
            "--out",
            str(output_dir),
            "--title",
            title,
            "--key",
            str(params.get("key", 0)),
        ]
        for parameter, option in (("speaker", "--spk"), ("language", "--lang"), ("steps", "--steps")):
            value = params.get(parameter)
            if value not in (None, ""):
                command.extend([option, str(value)])
        return command

    async def run(self, request: EngineRequest) -> EngineResult:
        if not (self.diffsinger_root / "scripts" / "infer.py").is_file():
            raise RuntimeError("DiffSinger scripts/infer.py was not found in the configured engine directory")
        job_dir = self.output_dir / f"diffsinger-{uuid.uuid4().hex}"
        job_dir.mkdir(parents=True, exist_ok=False)
        title = f"singing-{uuid.uuid4().hex}"
        command = self.build_command(request, job_dir, title)
        code, stdout, stderr = await self.executor(command, self.diffsinger_root, self.timeout_seconds)
        if code != 0:
            raise RuntimeError(f"DiffSinger inference failed ({code}): {(stderr or stdout).strip()}")
        candidates = sorted(job_dir.glob(f"{title}*.wav"))
        if len(candidates) != 1:
            raise RuntimeError("DiffSinger did not create exactly one WAV output")
        output = self.output_dir / candidates[0].name
        candidates[0].replace(output)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        try:
            info = sf.info(output)
        except (RuntimeError, sf.LibsndfileError) as error:
            output.unlink(missing_ok=True)
            raise RuntimeError("DiffSinger created an invalid WAV output") from error
        return EngineResult(
            output_audio=str(output),
            sample_rate=info.samplerate,
            engine=self.name,
            metadata={
                "duration": info.duration,
                "experiment": str(request.parameters["experiment"]),
                "source_format": "DiffSinger DS",
                "stdout": stdout[-2000:],
            },
        )

