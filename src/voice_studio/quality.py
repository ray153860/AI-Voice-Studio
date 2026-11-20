from pathlib import Path

import numpy as np
import soundfile as sf


def analyze_pitch(audio: np.ndarray, sample_rate: int) -> dict:
    mono = np.asarray(audio, dtype="float32").reshape(-1)
    frame_length = max(32, round(sample_rate * 0.05))
    hop_length = max(1, frame_length // 2)
    min_lag = max(1, int(sample_rate / 1000))
    max_lag = min(frame_length - 2, int(sample_rate / 60))
    if len(mono) < frame_length or max_lag <= min_lag:
        return {"voiced_ratio": 0.0, "pitch_median_hz": None, "pitch_low_hz": None, "pitch_high_hz": None, "range_semitones": None}
    pitches = []
    frame_count = 0
    window = np.hanning(frame_length).astype("float32")
    for start in range(0, len(mono) - frame_length + 1, hop_length):
        frame_count += 1
        frame = mono[start : start + frame_length]
        if float(np.sqrt(np.mean(np.square(frame)))) < 10 ** (-50 / 20):
            continue
        frame = (frame - np.mean(frame)) * window
        correlation = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        if correlation[0] <= 1e-12:
            continue
        candidates = correlation[min_lag : max_lag + 1]
        lag = min_lag + int(np.argmax(candidates))
        if correlation[lag] / correlation[0] < 0.3:
            continue
        pitches.append(sample_rate / lag)
    if not pitches:
        return {"voiced_ratio": 0.0, "pitch_median_hz": None, "pitch_low_hz": None, "pitch_high_hz": None, "range_semitones": None}
    low, median, high = np.percentile(pitches, [10, 50, 90])
    return {
        "voiced_ratio": len(pitches) / frame_count,
        "pitch_median_hz": float(median),
        "pitch_low_hz": float(low),
        "pitch_high_hz": float(high),
        "range_semitones": float(12 * np.log2(high / low)) if high > low else 0.0,
    }


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
        "pitch": analyze_pitch(np.mean(audio, axis=1), sample_rate),
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
