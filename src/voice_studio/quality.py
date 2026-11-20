from pathlib import Path

import numpy as np
import soundfile as sf


def analyze_audio(path: Path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if len(audio) == 0:
        raise ValueError("Audio is empty")
    absolute = np.abs(audio)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    frame_length = max(1, round(sample_rate * 0.02))
    padded_length = int(np.ceil(len(audio) / frame_length) * frame_length)
    padded = np.pad(audio, ((0, padded_length - len(audio)), (0, 0)))
    frames = padded.reshape(-1, frame_length, audio.shape[1])
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=(1, 2)))
    silent = frame_rms < 10 ** (-60 / 20)
    active_indices = np.flatnonzero(~silent)
    interior_silent_gaps = 0
    if len(active_indices) >= 2:
        interior = silent[active_indices[0] : active_indices[-1] + 1]
        run = 0
        for is_silent in interior:
            if is_silent:
                run += 1
            else:
                if run >= 3:
                    interior_silent_gaps += 1
                run = 0
    steps = np.abs(np.diff(audio, axis=0)) if len(audio) > 1 else np.zeros((1, audio.shape[1]))
    clipping_ratio = float(np.mean(absolute >= 0.999))
    return {
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "duration": len(audio) / sample_rate,
        "peak": peak,
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
        "dc_offset": [float(value) for value in np.mean(audio, axis=0)],
        "clipped_samples": int(np.sum(absolute >= 0.999)),
        "clipping_ratio": clipping_ratio,
        "silence_ratio": float(np.mean(silent)),
        "interior_silent_gaps": interior_silent_gaps,
        "max_sample_step": float(np.max(steps)),
        "flags": {
            "clipping_detected": clipping_ratio > 0,
            "possible_dropouts": interior_silent_gaps > 0,
            "mostly_silent": float(np.mean(silent)) > 0.9,
        },
        "interpretation": "Objective screening only; listening evaluation is still required.",
    }


def compare_audio(before_path: Path, after_path: Path):
    before = analyze_audio(before_path)
    after = analyze_audio(after_path)
    keys = ("peak", "rms_dbfs", "clipping_ratio", "silence_ratio", "interior_silent_gaps", "max_sample_step")
    return {
        "before": before,
        "after": after,
        "delta_after_minus_before": {key: after[key] - before[key] for key in keys},
        "interpretation": "Deltas describe signal changes, not a listening-quality verdict.",
    }
