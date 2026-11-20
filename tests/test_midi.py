import struct

import pytest

from voice_studio.midi import parse_midi


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
