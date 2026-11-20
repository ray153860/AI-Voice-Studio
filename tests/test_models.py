import hashlib

import pytest

from voice_studio.models import sha256_file, validate_model_filename


def test_model_filename_and_hash(tmp_path):
    assert validate_model_filename("folder/voice.pth", "rvc") == "voice.pth"
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    assert sha256_file(path) == hashlib.sha256(b"model").hexdigest()


def test_model_filename_rejects_wrong_format():
    with pytest.raises(ValueError):
        validate_model_filename("voice.zip", "singing")
