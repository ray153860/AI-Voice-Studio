import numpy as np
import pytest
import soundfile as sf

from voice_studio.engines import EngineResult
from voice_studio.workflows import run_cover_workflow


@pytest.mark.anyio
async def test_cover_workflow_sequences_separation_conversion_quality_and_mix(tmp_path):
    sample_rate = 8000
    song = tmp_path / "song.wav"
    vocals = tmp_path / "vocals.wav"
    accompaniment = tmp_path / "accompaniment.wav"
    converted = tmp_path / "converted.wav"
    output = tmp_path / "cover.wav"
    time = np.arange(sample_rate, dtype="float32") / sample_rate
    sf.write(song, np.sin(2 * np.pi * 220 * time) * 0.2, sample_rate)
    sf.write(vocals, np.sin(2 * np.pi * 220 * time) * 0.2, sample_rate)
    sf.write(accompaniment, np.sin(2 * np.pi * 110 * time) * 0.1, sample_rate)
    sf.write(converted, np.sin(2 * np.pi * 220 * time) * 0.25, sample_rate)
    calls = []

    class Separator:
        async def run(self, request):
            calls.append(("separate", request.input_audio))
            return EngineResult(
                str(vocals),
                sample_rate,
                "simulated-separator",
                artifacts={"vocals": str(vocals), "accompaniment": str(accompaniment)},
            )

    class Converter:
        async def run(self, request):
            calls.append(("convert", request.input_audio, request.parameters["model_path"]))
            return EngineResult(str(converted), sample_rate, "simulated-rvc")

    result = await run_cover_workflow(
        separator=Separator(),
        converter=Converter(),
        input_audio=song,
        output_audio=output,
        conversion_parameters={"model_path": "voice.pth", "pitch": 2},
    )

    assert calls == [("separate", str(song)), ("convert", str(vocals), "voice.pth")]
    assert output.is_file()
    assert result["stages"] == ["separation", "voice_conversion", "quality_comparison", "mix"]
    assert result["mix"]["duration"] == 1.0
    assert result["quality"]["before"]["pitch"]["pitch_median_hz"] is not None
    assert result["engines"] == {"separation": "simulated-separator", "voice_conversion": "simulated-rvc"}


@pytest.mark.anyio
async def test_cover_workflow_rejects_incomplete_separation_artifacts(tmp_path):
    input_audio = tmp_path / "song.wav"
    sf.write(input_audio, np.zeros(800, dtype="float32"), 8000)

    class IncompleteSeparator:
        async def run(self, request):
            return EngineResult("missing.wav", 8000, "separator", artifacts={"vocals": "missing.wav"})

    class UnusedConverter:
        async def run(self, request):
            raise AssertionError("converter must not run")

    with pytest.raises(RuntimeError, match="vocals and accompaniment"):
        await run_cover_workflow(
            IncompleteSeparator(),
            UnusedConverter(),
            input_audio,
            tmp_path / "output.wav",
            {"model_path": "voice.pth"},
        )

