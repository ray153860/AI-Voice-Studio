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
