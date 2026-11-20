import uuid
from pathlib import Path

import httpx
import soundfile as sf

from ..engines import EngineRequest, EngineResult


class GptSovitsAdapter:
    name = "gpt-sovits-api-v2"
    capability = "tts"

    def __init__(
        self,
        base_url: str,
        output_dir: Path,
        timeout_seconds: float = 300,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def run(self, request: EngineRequest) -> EngineResult:
        if not request.text or not request.reference_audio:
            raise ValueError("GPT-SoVITS requires text and a reference audio path")
        params = request.parameters
        payload = {
            "text": request.text,
            "text_lang": params.get("text_lang", "zh"),
            "ref_audio_path": request.reference_audio,
            "prompt_lang": params.get("prompt_lang", "zh"),
            "prompt_text": params.get("prompt_text", ""),
            "top_k": params.get("top_k", 15),
            "top_p": params.get("top_p", 1),
            "temperature": params.get("temperature", 1),
            "text_split_method": params.get("text_split_method", "cut5"),
            "speed_factor": params.get("speed_factor", 1.0),
            "media_type": "wav",
            "streaming_mode": False,
        }
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.post("/tts", json=payload)
        if response.status_code != 200:
            try:
                detail = response.json().get("message", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"GPT-SoVITS request failed ({response.status_code}): {detail}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"gpt-sovits-{uuid.uuid4().hex}.wav"
        output.write_bytes(response.content)
        try:
            info = sf.info(output)
        except (RuntimeError, sf.LibsndfileError) as error:
            output.unlink(missing_ok=True)
            raise RuntimeError("GPT-SoVITS returned an invalid WAV response") from error
        return EngineResult(
            output_audio=str(output),
            sample_rate=info.samplerate,
            engine=self.name,
            metadata={"duration": info.duration, "api": f"{self.base_url}/tts"},
        )
