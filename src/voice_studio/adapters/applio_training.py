import re
from pathlib import Path

from ..engines import TrainingRequest, TrainingResult
from .applio import ProcessExecutor, execute_process

MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ApplioTrainingAdapter:
    name = "applio-training-cli"
    capability = "voice_model_training"

    def __init__(
        self,
        applio_root: Path,
        python_executable: str,
        timeout_seconds: float = 86400,
        executor: ProcessExecutor = execute_process,
    ):
        self.applio_root = applio_root
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.executor = executor

    def build_commands(self, request: TrainingRequest) -> list[list[str]]:
        if not MODEL_NAME.fullmatch(request.model_name):
            raise ValueError("Applio model name must use 1-64 ASCII letters, numbers, underscores or hyphens")
        dataset = Path(request.dataset_path)
        if not dataset.is_dir():
            raise ValueError("Applio training dataset directory does not exist")
        params = request.parameters
        sample_rate = str(params.get("sample_rate", 40000))
        cpu_cores = str(params.get("cpu_cores", 4))
        gpu = str(params.get("gpu", 0))
        prefix = [self.python_executable, "core.py"]
        common = ["--model-name", request.model_name, "--sample-rate", sample_rate]
        return [
            prefix + ["preprocess", *common, "--dataset-path", str(dataset), "--cpu-cores", cpu_cores],
            prefix
            + [
                "extract",
                *common,
                "--cpu-cores",
                cpu_cores,
                "--gpu",
                gpu,
                "--f0-method",
                str(params.get("f0_method", "rmvpe")),
                "--embedder-model",
                str(params.get("embedder_model", "contentvec")),
            ],
            prefix
            + [
                "train",
                *common,
                "--gpu",
                gpu,
                "--batch-size",
                str(params.get("batch_size", 4)),
                "--save-every-epoch",
                str(params.get("save_every_epoch", 10)),
                "--total-epoch",
                str(params.get("total_epoch", 200)),
            ],
            prefix
            + [
                "index",
                "--model-name",
                request.model_name,
                "--index-algorithm",
                str(params.get("index_algorithm", "Auto")),
            ],
        ]

    async def run(self, request: TrainingRequest) -> TrainingResult:
        if not (self.applio_root / "core.py").is_file():
            raise RuntimeError("Applio core.py was not found in the configured engine directory")
        commands = self.build_commands(request)
        logs = []
        for command in commands:
            stage = command[2]
            code, stdout, stderr = await self.executor(command, self.applio_root, self.timeout_seconds)
            logs.append({"stage": stage, "stdout": stdout[-2000:]})
            if code != 0:
                raise RuntimeError(f"Applio {stage} failed ({code}): {(stderr or stdout).strip()}")
        output_dir = self.applio_root / "logs" / request.model_name
        model_candidates = sorted(output_dir.glob("*.pth"))
        index_candidates = sorted(output_dir.glob("*.index"))
        if not model_candidates or not index_candidates:
            raise RuntimeError("Applio training completed without both .pth and .index artifacts")
        return TrainingResult(
            engine=self.name,
            artifacts={"model": str(model_candidates[-1]), "index": str(index_candidates[-1])},
            metadata={"stages": [command[2] for command in commands], "logs": logs},
        )

