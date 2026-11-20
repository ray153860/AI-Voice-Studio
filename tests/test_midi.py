import struct

import pytest

from voice_studio.midi import align_lyrics_to_notes, parse_midi, tokenize_lyrics


def _track(events):
    return b"MTrk" + struct.pack(">I", len(events)) + events


def test_parse_midi_tempo_notes_and_rest(tmp_path):
    header = b"MThd" + struct.pack(">IHHH", 6, 1, 2, 480)
    tempo_track = _track(b"\x00\xff\x51\x03\x07\xa1\x20\x00\xff\x2f\x00")
    note_track = _track(
        b"\x00\x90\x3c\x64"
        b"\x83\x60\x80\x3c\x00"
        b"\x81\x70\x90\x3e\x64"
        b"\x83\x60\x80\x3e\x00"
        b"\x00\xff\x2f\x00"
    )
    path = tmp_path / "melody.mid"
    path.write_bytes(header + tempo_track + note_track)

    result = parse_midi(path)

    assert result["ticks_per_beat"] == 480
    assert result["tempos"][0]["bpm"] == 120
    assert [note["pitch"] for note in result["notes"]] == [60, 62]
    assert result["notes"][0]["duration_seconds"] == pytest.approx(0.5)
    assert result["rests"][0]["start_seconds"] == pytest.approx(0.5)
    assert result["rests"][0]["duration_seconds"] == pytest.approx(0.25)


def test_parse_midi_rejects_non_midi(tmp_path):
    path = tmp_path / "bad.mid"
    path.write_bytes(b"not midi")
    with pytest.raises(ValueError, match="Standard MIDI"):
        parse_midi(path)


def test_lyrics_align_to_monophonic_notes_with_slur_marker():
    notes = [
        {"pitch": 60, "start_tick": 0, "duration_ticks": 480, "start_seconds": 0.0, "duration_seconds": 0.5},
        {"pitch": 62, "start_tick": 480, "duration_ticks": 480, "start_seconds": 0.5, "duration_seconds": 0.5},
        {"pitch": 64, "start_tick": 960, "duration_ticks": 480, "start_seconds": 1.0, "duration_seconds": 0.5},
    ]

    result = align_lyrics_to_notes(notes, "你-好")

    assert tokenize_lyrics("Hello world 你好") == ["Hello", "world", "你", "好"]
    assert [item["lyric"] for item in result] == ["你", "", "好"]
    assert [item["slur"] for item in result] == [False, True, False]
    assert [item["pitch"] for item in result] == [60, 62, 64]


def test_lyrics_alignment_rejects_count_mismatch_and_polyphony():
    notes = [
        {"pitch": 60, "start_tick": 0, "duration_ticks": 480},
        {"pitch": 64, "start_tick": 0, "duration_ticks": 480},
    ]
    with pytest.raises(ValueError, match="monophonic"):
        align_lyrics_to_notes(notes, "你 好")

    with pytest.raises(ValueError, match="syllable count"):
        align_lyrics_to_notes([notes[0]], "你 好")
