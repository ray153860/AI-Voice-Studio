from voice_studio.diffsinger_ds import validate_diffsinger_ds


def test_ds_draft_reports_missing_model_phonemization_fields():
    report = validate_diffsinger_ds(
        [
            {
                "offset": 0.0,
                "text": "你 SP 好",
                "note_seq": "C4 rest D4",
                "note_dur": "0.5 0.25 0.5",
                "note_slur": "0 0 0",
            }
        ]
    )

    assert report["status"] == "valid_ds_draft_requires_model_phonemization"
    assert report["structurally_valid"] is True
    assert report["inference_fields_complete"] is False
    assert report["errors"] == []
    assert report["missing_fields"] == ["ph_seq", "ph_dur", "ph_num"]
    assert report["segments"] == [
        {
            "index": 0,
            "note_count": 3,
            "phoneme_count": 0,
            "duration_seconds": 1.25,
            "errors": [],
            "warnings": ["model-specific phonemization fields are missing: ph_seq, ph_dur, ph_num"],
        }
    ]


def test_ds_validation_rejects_mismatched_note_field_counts():
    report = validate_diffsinger_ds(
        [
            {
                "text": "你 好",
                "note_seq": "C4 D4",
                "note_dur": "0.5",
                "note_slur": "0 0",
            }
        ]
    )

    assert report["status"] == "invalid_ds"
    assert report["structurally_valid"] is False
    assert report["errors"] == ["segment 0: note_seq, note_dur, and note_slur must contain the same number of values"]
    assert report["segments"][0]["errors"] == [
        "note_seq, note_dur, and note_slur must contain the same number of values"
    ]


def test_ds_validation_marks_complete_phoneme_fields_without_claiming_model_compatibility():
    report = validate_diffsinger_ds(
        [
            {
                "text": "你",
                "note_seq": "C4",
                "note_dur": "0.5",
                "note_slur": "0",
                "ph_seq": "n i",
                "ph_dur": "0.2 0.3",
                "ph_num": "2",
            }
        ]
    )

    assert report["status"] == "fields_complete_model_compatibility_unverified"
    assert report["structurally_valid"] is True
    assert report["inference_fields_complete"] is True
    assert report["missing_fields"] == []
    assert report["segments"][0]["phoneme_count"] == 2
    assert report["segments"][0]["warnings"] == [
        "phoneme fields are structurally complete; target model dictionary compatibility is unverified"
    ]


def test_ds_validation_reports_empty_document_instead_of_crashing():
    report = validate_diffsinger_ds([])

    assert report == {
        "status": "invalid_ds",
        "structurally_valid": False,
        "inference_fields_complete": False,
        "errors": ["DS document must contain at least one segment"],
        "missing_fields": ["ph_seq", "ph_dur", "ph_num"],
        "segments": [],
    }


def test_ds_validation_rejects_invalid_pitch_duration_and_slur_values():
    report = validate_diffsinger_ds(
        [{"text": "你", "note_seq": "H4", "note_dur": "-0.5", "note_slur": "2"}]
    )

    assert report["status"] == "invalid_ds"
    assert report["segments"][0]["errors"] == [
        "note_seq contains an invalid note: H4",
        "note_dur values must be finite positive numbers",
        "note_slur values must be 0 or 1",
    ]


def test_ds_validation_rejects_inconsistent_phoneme_fields():
    report = validate_diffsinger_ds(
        [
            {
                "text": "你",
                "note_seq": "C4",
                "note_dur": "0.5",
                "note_slur": "0",
                "ph_seq": "n i",
                "ph_dur": "0.5",
                "ph_num": "1",
            }
        ]
    )

    assert report["status"] == "invalid_ds"
    assert report["inference_fields_complete"] is False
    assert report["segments"][0]["errors"] == [
        "ph_seq and ph_dur must contain the same number of values",
        "ph_num values must sum to the ph_seq value count",
    ]


def test_ds_validation_checks_every_segment():
    complete = {
        "text": "你",
        "note_seq": "C4",
        "note_dur": "0.5",
        "note_slur": "0",
        "ph_seq": "n i",
        "ph_dur": "0.2 0.3",
        "ph_num": "2",
    }
    draft = {"text": "好", "note_seq": "D4", "note_dur": "0.5", "note_slur": "0"}

    report = validate_diffsinger_ds([complete, draft])

    assert report["status"] == "valid_ds_draft_requires_model_phonemization"
    assert report["missing_fields"] == ["ph_seq", "ph_dur", "ph_num"]
    assert [segment["index"] for segment in report["segments"]] == [0, 1]
    assert [segment["phoneme_count"] for segment in report["segments"]] == [2, 0]


def test_ds_validation_requires_score_fields():
    report = validate_diffsinger_ds({})

    assert report["status"] == "invalid_ds"
    assert report["segments"][0]["errors"] == [
        "text, note_seq, note_dur, and note_slur are required non-empty strings"
    ]


def test_ds_validation_rejects_invalid_phoneme_durations_and_per_note_counts():
    report = validate_diffsinger_ds(
        [
            {
                "text": "你 好",
                "note_seq": "C4 D4",
                "note_dur": "0.5 0.5",
                "note_slur": "0 0",
                "ph_seq": "n i h ao",
                "ph_dur": "0.2 0.3 nope -0.1",
                "ph_num": "4",
            }
        ]
    )

    assert report["segments"][0]["errors"] == [
        "ph_dur values must be finite positive numbers",
        "ph_num must contain one value for each note",
    ]
