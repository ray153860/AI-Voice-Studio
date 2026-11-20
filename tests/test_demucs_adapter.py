import numpy as np
import pytest
import soundfile as sf

from voice_studio.adapters.demucs import DemucsAdapter
from voice_studio.engines import EngineRequest


@pytest.mark.anyio
async def test_demucs_contract_with_simulated_process(tmp_path):
    input_audio = tmp_path / "song.wav"
    sf.write(input_audio, np.zeros(4410, dtype="float32"), 44100)
    captured = {}

    async def executor(command, cwd, timeout):
        captured["command"] = list(command)
        output_root = __import__("pathlib").Path(command[command.index("-o") + 1])
        stem_dir = output_root / "htdemucs" / "song"
        stem_dir.mkdir(parents=True)
        sf.write(stem_dir / "vocals.wav", np.zeros(4410, dtype="float32"), 44100)
        sf.write(stem_dir / "no_vocals.wav", np.zeros(4410, dtype="float32"), 44100)
        return 0, "Separated", ""

    adapter = DemucsAdapter("demucs-engine", tmp_path / "outputs", executor=executor)
    result = await adapter.run(EngineRequest(input_audio=str(input_audio)))

    assert captured["command"][0:2] == ["demucs-engine", "--two-stems=vocals"]
    assert result.sample_rate == 44100
    assert set(result.artifacts) == {"vocals", "accompaniment"}
    assert all(__import__("pathlib").Path(path).is_file() for path in result.artifacts.values())


@pytest.mark.anyio
async def test_demucs_requires_input(tmp_path):
    adapter = DemucsAdapter("demucs", tmp_path)
    with pytest.raises(ValueError, match="input audio"):
        await adapter.run(EngineRequest())
