import hashlib
from pathlib import Path

MODEL_EXTENSIONS = {
    "rvc": {".pth", ".index"},
    "gpt-sovits": {".ckpt", ".pth"},
    "singing": {".ckpt", ".onnx", ".pt", ".pth"},
}


def validate_model_filename(filename: str, model_type: str) -> str:
    if model_type not in MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported model type: {model_type}")
    safe_name = Path(filename).name
    if not safe_name or Path(safe_name).suffix.lower() not in MODEL_EXTENSIONS[model_type]:
        allowed = ", ".join(sorted(MODEL_EXTENSIONS[model_type]))
        raise ValueError(f"{model_type} model must use one of: {allowed}")
    return safe_name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
