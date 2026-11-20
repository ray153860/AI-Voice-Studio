import numpy as np
import soundfile as sf

from voice_studio.audio import inspect_audio, mix_audio, process_audio


def test_real_wav_workflow(tmp_path):
    src=tmp_path/"in.wav"; dst=tmp_path/"out.wav"
    sf.write(src, (np.sin(np.arange(24000)*.04)*.2).astype("float32"), 24000)
    assert inspect_audio(src)["duration"] == 1.0
    result=process_audio(src,dst)
    data,sr=sf.read(dst)
    assert sr == 24000 and len(data)==24000 and result["peak"] <= .981


def test_real_mix_aligns_lengths_and_prevents_clipping(tmp_path):
    vocals = tmp_path / "vocals.wav"
    accompaniment = tmp_path / "accompaniment.wav"
    output = tmp_path / "mix.wav"
    sf.write(vocals, np.ones((8000, 2), dtype="float32") * 0.8, 8000)
    sf.write(accompaniment, np.ones((4000, 2), dtype="float32") * 0.8, 8000)

    result = mix_audio(vocals, accompaniment, output)
    mixed, sample_rate = sf.read(output, always_2d=True)

    assert sample_rate == 8000
    assert len(mixed) == 8000
    assert result["duration"] == 1.0
    assert result["peak_before_limiter"] > 1
    assert np.max(np.abs(mixed)) <= 0.981
