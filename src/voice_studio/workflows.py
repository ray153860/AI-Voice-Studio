import asyncio
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import mix_audio
from .engines import Engine, EngineRequest
from .quality import compare_audio


async def run_chunked_conversion(
    converter: Engine,
    input_audio: Path,
    output_audio: Path,
    conversion_parameters: dict,
    chunk_seconds: float,
    overlap_seconds: float,
    workspace: Path,
) -> dict:
    audio, sample_rate = sf.read(input_audio, dtype="float32", always_2d=True)
    chunk_samples = round(chunk_seconds * sample_rate)
    overlap_samples = round(overlap_seconds * sample_rate)
    if chunk_samples < 2 or overlap_samples < 1 or overlap_samples >= chunk_samples:
        raise ValueError("chunk and overlap durations must produce 0 < overlap < chunk")
    workspace.mkdir(parents=True, exist_ok=True)
    step = chunk_samples - overlap_samples
    starts = [0]
    while starts[-1] + chunk_samples < len(audio):
        starts.append(starts[-1] + step)
    accumulator = np.zeros_like(audio, dtype="float64")
    weights = np.zeros((len(audio), 1), dtype="float64")
    engine_names = []
    for number, start in enumerate(starts):
        end = min(start + chunk_samples, len(audio))
        source_chunk = workspace / f"source-{number:04d}.wav"
        sf.write(source_chunk, audio[start:end], sample_rate, subtype="FLOAT")
        result = await converter.run(
            EngineRequest(input_audio=str(source_chunk), parameters=dict(conversion_parameters))
        )
        converted, converted_rate = sf.read(result.output_audio, dtype="float32", always_2d=True)
        if converted_rate != sample_rate or converted.shape != audio[start:end].shape:
            raise RuntimeError("Chunked conversion must preserve sample rate, channels and frame count")
        chunk_weights = np.ones((len(converted), 1), dtype="float64")
        fade = min(overlap_samples, len(converted))
        if start > 0:
            chunk_weights[:fade, 0] = np.linspace(0, 1, fade, endpoint=False)
        if end < len(audio):
            chunk_weights[-fade:, 0] *= np.linspace(1, 0, fade, endpoint=False)
        accumulator[start:end] += converted * chunk_weights
        weights[start:end] += chunk_weights
        engine_names.append(result.engine)
    stitched = (accumulator / np.maximum(weights, 1e-12)).astype("float32")
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio, stitched, sample_rate, subtype="PCM_16")
    boundary_steps = [
        float(np.max(np.abs(stitched[start] - stitched[start - 1])))
        for start in starts[1:]
        if 0 < start < len(stitched)
    ]
    return {
        "chunk_count": len(starts),
        "chunk_samples": chunk_samples,
        "overlap_samples": overlap_samples,
        "boundary_max_step": max(boundary_steps, default=0.0),
        "engines": sorted(set(engine_names)),
    }


async def run_cover_workflow(
    separator: Engine,
    converter: Engine,
    input_audio: Path,
    output_audio: Path,
    conversion_parameters: dict,
    separation_parameters: dict | None = None,
    vocal_gain_db: float = 0,
    accompaniment_gain_db: float = 0,
    chunk_seconds: float = 0,
    overlap_seconds: float = 1,
    workspace: Path | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict:
    separation = await separator.run(
        EngineRequest(input_audio=str(input_audio), parameters=separation_parameters or {})
    )
    if not {"vocals", "accompaniment"}.issubset(separation.artifacts):
        raise RuntimeError("Separation must provide both vocals and accompaniment artifacts")
    vocals = Path(separation.artifacts["vocals"])
    accompaniment = Path(separation.artifacts["accompaniment"])
    if not vocals.is_file() or not accompaniment.is_file():
        raise RuntimeError("Separation vocals or accompaniment artifact is missing")
    if progress_callback:
        progress_callback("separation", 25)
    chunking = None
    if chunk_seconds > 0:
        if workspace is None:
            raise ValueError("chunked cover conversion requires a workspace")
        converted = workspace / "stitched-converted-vocals.wav"
        chunking = await run_chunked_conversion(
            converter,
            vocals,
            converted,
            conversion_parameters,
            chunk_seconds,
            overlap_seconds,
            workspace,
        )
        conversion_engine = ",".join(chunking["engines"])
    else:
        conversion = await converter.run(
            EngineRequest(input_audio=str(vocals), parameters=conversion_parameters)
        )
        converted = Path(conversion.output_audio)
        conversion_engine = conversion.engine
    if not converted.is_file():
        raise RuntimeError("Voice conversion output is missing")
    if progress_callback:
        progress_callback("voice_conversion", 55)
    quality = await asyncio.to_thread(compare_audio, vocals, converted)
    if progress_callback:
        progress_callback("quality_comparison", 75)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    mix = await asyncio.to_thread(
        mix_audio,
        converted,
        accompaniment,
        output_audio,
        vocal_gain_db,
        accompaniment_gain_db,
    )
    if progress_callback:
        progress_callback("mix", 100)
    return {
        "stages": ["separation", "voice_conversion", "quality_comparison", "mix"],
        "engines": {"separation": separation.engine, "voice_conversion": conversion_engine},
        "chunking": chunking,
        "quality": quality,
        "mix": mix,
        "artifacts": {
            "source_vocals": str(vocals),
            "accompaniment": str(accompaniment),
            "converted_vocals": str(converted),
            "mixed_output": str(output_audio),
        },
    }

