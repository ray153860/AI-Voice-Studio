import numpy as np
import soundfile as sf

from voice_studio.audio import inspect_audio, process_audio


def test_real_wav_workflow(tmp_path):
    src=tmp_path/"in.wav"; dst=tmp_path/"out.wav"
    sf.write(src, (np.sin(np.arange(24000)*.04)*.2).astype("float32"), 24000)
    assert inspect_audio(src)["duration"] == 1.0
    result=process_audio(src,dst)
    data,sr=sf.read(dst)
    assert sr == 24000 and len(data)==24000 and result["peak"] <= .981
