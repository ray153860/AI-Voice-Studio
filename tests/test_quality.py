import numpy as np
import soundfile as sf

from voice_studio.quality import analyze_audio, compare_audio


def test_audio_quality_metrics_detect_gap_and_clipping(tmp_path):
    sample_rate = 1000
    audio = np.ones(sample_rate, dtype="float32") * 0.2
    audio[300:400] = 0
    audio[700:710] = 1
    path = tmp_path / "quality.wav"
    sf.write(path, audio, sample_rate, subtype="FLOAT")

    result = analyze_audio(path)

    assert result["duration"] == 1
    assert result["clipped_samples"] == 10
    assert result["interior_silent_gaps"] == 1
    assert result["flags"]["clipping_detected"] is True
    assert result["flags"]["possible_dropouts"] is True


def test_compare_audio_reports_deltas_without_quality_claim(tmp_path):
    before = tmp_path / "before.wav"
    after = tmp_path / "after.wav"
    sf.write(before, np.ones(1000, dtype="float32"), 1000, subtype="FLOAT")
    sf.write(after, np.ones(1000, dtype="float32") * 0.5, 1000, subtype="FLOAT")

    result = compare_audio(before, after)

    assert result["delta_after_minus_before"]["clipping_ratio"] == -1
    assert result["delta_after_minus_before"]["peak"] == -0.5
    assert "not a listening-quality verdict" in result["interpretation"]
