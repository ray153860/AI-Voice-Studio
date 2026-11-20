import numpy as np
import pytest
import soundfile as sf

from voice_studio.adapters.applio import ApplioAdapter
from voice_studio.engines import EngineRequest


@pytest.mark.anyio
async def test_applio_cli_contract_with_simulated_process(tmp_path):
    root = tmp_path / "applio"
    root.mkdir()
    (root / "core.py").write_text("# test placeholder")
    input_audio = tmp_path / "input.wav"
    model = tmp_path / "voice.pth"
    sf.write(input_audio, np.zeros(1600, dtype="float32"), 16000)
    model.write_bytes(b"model")
    captured = {}

    async def executor(command, cwd, timeout):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        output = command[command.index("--output-path") + 1]
        sf.write(output, np.zeros(4800, dtype="float32"), 48000)
        return 0, "Success", ""

    adapter = ApplioAdapter(root, "engine-python", tmp_path / "outputs", executor=executor)
    result = await adapter.run(
        EngineRequest(
            input_audio=str(input_audio),
            parameters={"model_path": str(model), "pitch": 2, "f0_method": "rmvpe"},
        )
    )

    assert captured["command"][:3] == ["engine-python", "core.py", "infer"]
    assert captured["command"][captured["command"].index("--pitch") + 1] == "2"
    assert captured["cwd"] == root
    assert result.sample_rate == 48000
    assert result.engine == "applio-cli"


@pytest.mark.anyio
async def test_applio_adapter_requires_external_engine(tmp_path):
    adapter = ApplioAdapter(tmp_path / "missing", "python", tmp_path / "outputs")
    with pytest.raises(RuntimeError, match="core.py"):
        await adapter.run(EngineRequest(input_audio="input.wav", parameters={"model_path": "voice.pth"}))
