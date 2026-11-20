from pathlib import Path

import numpy as np
import soundfile as sf


def inspect_audio(path: Path):
    info = sf.info(path)
    if info.frames < 1 or info.samplerate < 1:
        raise ValueError("Audio is empty or invalid")
    return {"sample_rate": info.samplerate, "channels": info.channels, "duration": info.duration, "format": info.format}

def process_audio(source: Path, target: Path):
    data, sr = sf.read(source, dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(data)))
    if peak == 0:
        raise ValueError("Cannot normalize silent audio")
    data = data * min(0.98 / peak, 1.0)
    fade = min(int(sr * .01), len(data) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        data[:fade] *= ramp
        data[-fade:] *= ramp[::-1]
    sf.write(target, data, sr, subtype="PCM_16")
    return {"sample_rate": sr, "duration": len(data) / sr, "peak": float(np.max(np.abs(data)))}


def mix_audio(
    vocals_path: Path,
    accompaniment_path: Path,
    target: Path,
    vocal_gain_db: float = 0,
    accompaniment_gain_db: float = 0,
):
    vocals, vocals_sr = sf.read(vocals_path, dtype="float32", always_2d=True)
    accompaniment, accompaniment_sr = sf.read(accompaniment_path, dtype="float32", always_2d=True)
    if vocals_sr != accompaniment_sr:
        raise ValueError("Vocal and accompaniment sample rates must match")
    if vocals.shape[1] != accompaniment.shape[1]:
        raise ValueError("Vocal and accompaniment channel counts must match")
    length = max(len(vocals), len(accompaniment))
    vocals = np.pad(vocals, ((0, length - len(vocals)), (0, 0)))
    accompaniment = np.pad(accompaniment, ((0, length - len(accompaniment)), (0, 0)))
    mixed = vocals * (10 ** (vocal_gain_db / 20)) + accompaniment * (10 ** (accompaniment_gain_db / 20))
    source_peak = float(np.max(np.abs(mixed)))
    limiter_gain = min(0.98 / source_peak, 1.0) if source_peak else 1.0
    mixed *= limiter_gain
    sf.write(target, mixed, vocals_sr, subtype="PCM_16")
    return {
        "sample_rate": vocals_sr,
        "duration": length / vocals_sr,
        "peak_before_limiter": source_peak,
        "limiter_gain": limiter_gain,
        "peak": float(np.max(np.abs(mixed))),
    }
