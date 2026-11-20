# Changelog

## 0.1.0 - 2026-11-20

- Initial local workflow UI, WAV validation and normalization workflow, persistent job tracking, retry/cancellation, model registry and engine adapter contracts.
- Added GPT-SoVITS API, Applio RVC, Demucs separation, and DiffSinger DS acoustic-inference adapter contracts; all AI adapters remain explicitly unverified against real models.
- Added real WAV mixing, MIDI inspection, objective audio quality analysis, and Linux/Windows CI.
- Added an Applio preprocess/extract/train/index adapter contract without claiming real training execution.
- Added validated multi-file training dataset registration, RVC training jobs, and automatic unverified artifact registration.
- Added lightweight monophonic pitch-range screening for objective singing and conversion checks.
- Added the preset cover workflow: separation, RVC conversion, objective before/after comparison, and accompaniment mix.
- Added optional overlap-add chunked conversion for long vocals with exact length and boundary-step checks.
- Replaced the deprecated Starlette TestClient fallback with direct HTTPX ASGI transport and made CI fail on warnings.
- Added a Windows PyInstaller one-folder build, packaged API smoke test, and unsigned artifact upload workflow.
