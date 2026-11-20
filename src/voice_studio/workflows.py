import asyncio
from pathlib import Path

from .audio import mix_audio
from .engines import Engine, EngineRequest
from .quality import compare_audio


async def run_cover_workflow(
    separator: Engine,
    converter: Engine,
    input_audio: Path,
    output_audio: Path,
    conversion_parameters: dict,
    separation_parameters: dict | None = None,
    vocal_gain_db: float = 0,
    accompaniment_gain_db: float = 0,
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
    conversion = await converter.run(
        EngineRequest(input_audio=str(vocals), parameters=conversion_parameters)
    )
    converted = Path(conversion.output_audio)
    if not converted.is_file():
        raise RuntimeError("Voice conversion output is missing")
    quality = await asyncio.to_thread(compare_audio, vocals, converted)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    mix = await asyncio.to_thread(
        mix_audio,
        converted,
        accompaniment,
        output_audio,
        vocal_gain_db,
        accompaniment_gain_db,
    )
    return {
        "stages": ["separation", "voice_conversion", "quality_comparison", "mix"],
        "engines": {"separation": separation.engine, "voice_conversion": conversion.engine},
        "quality": quality,
        "mix": mix,
        "artifacts": {
            "source_vocals": str(vocals),
            "accompaniment": str(accompaniment),
            "converted_vocals": str(converted),
            "mixed_output": str(output_audio),
        },
    }

