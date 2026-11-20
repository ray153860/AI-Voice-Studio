from pathlib import Path

import pytest

from voice_studio.adapters.applio_training import ApplioTrainingAdapter
from voice_studio.engines import TrainingRequest


@pytest.mark.anyio
async def test_applio_training_runs_official_stages_and_collects_artifacts(tmp_path):
    root = tmp_path / "applio"
    root.mkdir()
    (root / "core.py").write_text("# test placeholder")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "clip.wav").write_bytes(b"test fixture")
    commands = []

    async def executor(command, cwd, timeout):
        commands.append(list(command))
        stage = command[2]
        if stage == "train":
            output = root / "logs" / "DemoVoice"
            output.mkdir(parents=True, exist_ok=True)
            (output / "DemoVoice.pth").write_bytes(b"model")
            (output / "G_20.pth").write_bytes(b"training checkpoint")
        if stage == "index":
            (root / "logs" / "DemoVoice" / "DemoVoice.index").write_bytes(b"index")
        return 0, f"{stage} complete", ""

    adapter = ApplioTrainingAdapter(root, "engine-python", executor=executor)
    result = await adapter.run(
        TrainingRequest(
            model_name="DemoVoice",
            dataset_path=str(dataset),
            parameters={"gpu": "0", "total_epoch": 20, "batch_size": 2},
        )
    )

    assert [command[2] for command in commands] == ["preprocess", "extract", "train", "index"]
    assert commands[0][commands[0].index("--dataset-path") + 1] == str(dataset)
    assert commands[2][commands[2].index("--total-epoch") + 1] == "20"
    assert result.engine == "applio-training-cli"
    assert set(result.artifacts) == {"model", "index"}
    assert all(Path(path).is_file() for path in result.artifacts.values())
    assert Path(result.artifacts["model"]).name == "DemoVoice.pth"
    assert result.metadata["stages"] == ["preprocess", "extract", "train", "index"]


@pytest.mark.anyio
async def test_applio_training_rejects_unsafe_model_name(tmp_path):
    root = tmp_path / "applio"
    root.mkdir()
    (root / "core.py").write_text("# test placeholder")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    adapter = ApplioTrainingAdapter(root, "python")

    with pytest.raises(ValueError, match="model name"):
        await adapter.run(TrainingRequest(model_name="../escape", dataset_path=str(dataset)))

