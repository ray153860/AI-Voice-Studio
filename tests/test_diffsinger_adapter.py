from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_studio.adapters.diffsinger import DiffSingerAdapter
from voice_studio.engines import EngineRequest


@pytest.mark.anyio
async def test_diffsinger_acoustic_contract_with_simulated_process(tmp_path):
    root = tmp_path / "diffsinger"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "infer.py").write_text("# test placeholder")
    ds_file = tmp_path / "song.ds"
    ds_file.write_text('[{"ph_seq": "a", "ph_dur": "1.0", "f0_seq": "440"}]')
    captured = {}

    async def executor(command, cwd, timeout):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        output_dir = Path(command[command.index("--out") + 1])
        title = command[command.index("--title") + 1]
        sf.write(output_dir / f"{title}.wav", np.zeros(4410, dtype="float32"), 44100)
        return 0, "Synthesis complete", ""

    adapter = DiffSingerAdapter(root, "engine-python", tmp_path / "outputs", executor=executor)
    result = await adapter.run(
        EngineRequest(
            parameters={
                "ds_path": str(ds_file),
                "experiment": "licensed-singer",
                "speaker": "singer-a",
                "key": 2,
            }
        )
    )

    assert captured["command"][:4] == ["engine-python", "scripts/infer.py", "acoustic", str(ds_file)]
    assert captured["command"][captured["command"].index("--exp") + 1] == "licensed-singer"
    assert captured["command"][captured["command"].index("--spk") + 1] == "singer-a"
    assert captured["command"][captured["command"].index("--key") + 1] == "2"
    assert captured["cwd"] == root
    assert result.sample_rate == 44100
    assert result.engine == "diffsinger-cli"
    assert Path(result.output_audio).is_file()
    assert Path(result.output_audio).parent == tmp_path / "outputs"


@pytest.mark.anyio
async def test_diffsinger_requires_ds_file_and_experiment(tmp_path):
    root = tmp_path / "diffsinger"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "infer.py").write_text("# test placeholder")
    adapter = DiffSingerAdapter(root, "python", tmp_path / "outputs")

    with pytest.raises(ValueError, match="DS file"):
        await adapter.run(EngineRequest(parameters={"experiment": "voice"}))

    ds_file = tmp_path / "song.ds"
    ds_file.write_text("[]")
    with pytest.raises(ValueError, match="experiment"):
        await adapter.run(EngineRequest(parameters={"ds_path": str(ds_file)}))

