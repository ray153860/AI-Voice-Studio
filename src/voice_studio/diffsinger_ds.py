import math
import re

MODEL_PHONEMIZATION_FIELDS = ("ph_seq", "ph_dur", "ph_num")
NOTE_VALUE = re.compile(r"(?:rest|[A-G](?:#|b)?-?\d+)(?:/(?:rest|[A-G](?:#|b)?-?\d+))*$")


def _empty_report(error: str) -> dict:
    return {
        "status": "invalid_ds",
        "structurally_valid": False,
        "inference_fields_complete": False,
        "errors": [error],
        "missing_fields": list(MODEL_PHONEMIZATION_FIELDS),
        "segments": [],
    }


def validate_diffsinger_ds(document: object) -> dict:
    segments = [document] if isinstance(document, dict) else document
    if not isinstance(segments, list) or not segments:
        return _empty_report("DS document must contain at least one segment")

    reports = []
    errors = []
    missing_fields = set()
    for index, segment in enumerate(segments):
        segment_errors = []
        if not isinstance(segment, dict):
            segment_errors.append("segment must be a JSON object")
            notes = []
            durations = []
            slurs = []
            phonemes = []
            missing = list(MODEL_PHONEMIZATION_FIELDS)
        else:
            score_fields = ("text", "note_seq", "note_dur", "note_slur")
            score_fields_valid = all(
                isinstance(segment.get(field), str) and segment[field].strip() for field in score_fields
            )
            if not score_fields_valid:
                segment_errors.append("text, note_seq, note_dur, and note_slur are required non-empty strings")
            notes = str(segment.get("note_seq", "")).split()
            try:
                durations = [float(value) for value in str(segment.get("note_dur", "")).split()]
            except ValueError:
                durations = []
                segment_errors.append("note_dur values must be finite positive numbers")
            slurs = str(segment.get("note_slur", "")).split()
            if score_fields_valid and (len(notes) != len(durations) or len(notes) != len(slurs)):
                segment_errors.append("note_seq, note_dur, and note_slur must contain the same number of values")
            invalid_note = next((note for note in notes if not NOTE_VALUE.fullmatch(note)), None)
            if invalid_note:
                segment_errors.append(f"note_seq contains an invalid note: {invalid_note}")
            if durations and any(not math.isfinite(value) or value <= 0 for value in durations):
                segment_errors.append("note_dur values must be finite positive numbers")
            if any(value not in ("0", "1") for value in slurs):
                segment_errors.append("note_slur values must be 0 or 1")

            missing = [field for field in MODEL_PHONEMIZATION_FIELDS if not segment.get(field)]
            phonemes = str(segment.get("ph_seq", "")).split()
            if not missing:
                phoneme_duration_values = str(segment["ph_dur"]).split()
                try:
                    phoneme_durations = [float(value) for value in phoneme_duration_values]
                except ValueError:
                    phoneme_durations = []
                    segment_errors.append("ph_dur values must be finite positive numbers")
                try:
                    phoneme_counts = [int(value) for value in str(segment["ph_num"]).split()]
                except ValueError:
                    phoneme_counts = []
                    segment_errors.append("ph_num values must be integers")
                if phoneme_durations and any(
                    not math.isfinite(value) or value <= 0 for value in phoneme_durations
                ):
                    segment_errors.append("ph_dur values must be finite positive numbers")
                if len(phonemes) != len(phoneme_duration_values):
                    segment_errors.append("ph_seq and ph_dur must contain the same number of values")
                if len(phoneme_counts) != len(notes):
                    segment_errors.append("ph_num must contain one value for each note")
                if phoneme_counts and sum(phoneme_counts) != len(phonemes):
                    segment_errors.append("ph_num values must sum to the ph_seq value count")

        missing_fields.update(missing)
        if missing:
            warnings = [f"model-specific phonemization fields are missing: {', '.join(missing)}"]
        else:
            warnings = [
                "phoneme fields are structurally complete; target model dictionary compatibility is unverified"
            ]
        reports.append(
            {
                "index": index,
                "note_count": len(notes),
                "phoneme_count": len(phonemes),
                "duration_seconds": round(sum(durations), 6),
                "errors": segment_errors,
                "warnings": warnings,
            }
        )
        errors.extend(f"segment {index}: {error}" for error in segment_errors)

    ordered_missing = [field for field in MODEL_PHONEMIZATION_FIELDS if field in missing_fields]
    complete = not errors and not ordered_missing
    status = (
        "invalid_ds"
        if errors
        else "fields_complete_model_compatibility_unverified"
        if complete
        else "valid_ds_draft_requires_model_phonemization"
    )
    return {
        "status": status,
        "structurally_valid": not errors,
        "inference_fields_complete": complete,
        "errors": errors,
        "missing_fields": ordered_missing,
        "segments": reports,
    }
