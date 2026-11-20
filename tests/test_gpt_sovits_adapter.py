import io

import httpx
import numpy as np
import pytest
import soundfile as sf

from voice_studio.adapters.gpt_sovits import GptSovitsAdapter
from voice_studio.engines import EngineRequest


@pytest.mark.anyio
async def test_gpt_sovits_adapter_contract(tmp_path):
    audio = io.BytesIO()
    sf.write(audio, np.zeros(2400, dtype="float32"), 24000, format="WAV")

    async def handler(request):
        payload = __import__("json").loads(request.content)
        assert request.url.path == "/tts"
        assert payload["text"] == "测试文本"
        assert payload["ref_audio_path"] == "shared/reference.wav"
        return httpx.Response(200, content=audio.getvalue(), headers={"content-type": "audio/wav"})

    adapter = GptSovitsAdapter(
        "http://engine.test:9880",
        tmp_path,
        transport=httpx.MockTransport(handler),
    )
    result = await adapter.run(EngineRequest(text="测试文本", reference_audio="shared/reference.wav"))

    assert result.engine == "gpt-sovits-api-v2"
    assert result.sample_rate == 24000
    assert result.metadata["duration"] == pytest.approx(0.1)


@pytest.mark.anyio
async def test_gpt_sovits_adapter_reports_upstream_error(tmp_path):
    async def handler(request):
        return httpx.Response(400, json={"message": "model not loaded"})

    adapter = GptSovitsAdapter("http://engine.test", tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="model not loaded"):
        await adapter.run(EngineRequest(text="text", reference_audio="reference.wav"))


@pytest.mark.anyio
async def test_gpt_sovits_adapter_requires_inputs(tmp_path):
    adapter = GptSovitsAdapter("http://engine.test", tmp_path)
    with pytest.raises(ValueError, match="reference audio"):
        await adapter.run(EngineRequest(text="text"))
