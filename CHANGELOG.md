# Changelog

## 0.1.0 - 2026-11-20

- Initial local workflow UI, WAV validation and normalization workflow, persistent job tracking, retry/cancellation, model registry and engine adapter contracts.
- Added GPT-SoVITS API, Applio RVC, Demucs separation, and DiffSinger DS acoustic-inference adapter contracts; all AI adapters remain explicitly unverified against real models.
- Added real WAV mixing, MIDI inspection, objective audio quality analysis, and Linux/Windows CI.
- Added an Applio preprocess/extract/train/index adapter contract without claiming real training execution.
- Added validated multi-file training dataset registration, RVC training jobs, and automatic unverified artifact registration.
