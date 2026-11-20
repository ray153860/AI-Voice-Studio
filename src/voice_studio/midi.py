import struct
from collections import defaultdict
from pathlib import Path


def _variable_length(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if offset >= len(data):
            raise ValueError("Truncated MIDI variable-length value")
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset
    raise ValueError("Invalid MIDI variable-length value")


def _parse_track(data: bytes, track_number: int):
    offset = 0
    tick = 0
    running_status = None
    events = []
    tempos = []
    lyrics = []
    order = 0
    while offset < len(data):
        delta, offset = _variable_length(data, offset)
        tick += delta
        if offset >= len(data):
            raise ValueError("Truncated MIDI event")
        first = data[offset]
        if first < 0x80:
            if running_status is None:
                raise ValueError("MIDI running status used before a channel event")
            status = running_status
        else:
            status = first
            offset += 1
        if status == 0xFF:
            running_status = None
            if offset >= len(data):
                raise ValueError("Truncated MIDI meta event")
            meta_type = data[offset]
            offset += 1
            length, offset = _variable_length(data, offset)
            payload = data[offset : offset + length]
            if len(payload) != length:
                raise ValueError("Truncated MIDI meta payload")
            offset += length
            if meta_type == 0x51 and length == 3:
                tempos.append((tick, int.from_bytes(payload, "big"), track_number, order))
            elif meta_type in (0x05, 0x01):
                lyrics.append({"tick": tick, "text": payload.decode("utf-8", errors="replace")})
            if meta_type == 0x2F:
                break
        elif status in (0xF0, 0xF7):
            running_status = None
            length, offset = _variable_length(data, offset)
            offset += length
            if offset > len(data):
                raise ValueError("Truncated MIDI system-exclusive event")
        else:
            running_status = status
            event_type = status & 0xF0
            channel = status & 0x0F
            data_length = 1 if event_type in (0xC0, 0xD0) else 2
            payload = data[offset : offset + data_length]
            if len(payload) != data_length:
                raise ValueError("Truncated MIDI channel event")
            offset += data_length
            if event_type in (0x80, 0x90):
                note, velocity = payload
                kind = "off" if event_type == 0x80 or velocity == 0 else "on"
                events.append((tick, track_number, order, kind, channel, note, velocity))
        order += 1
    return events, tempos, lyrics


def parse_midi(path: Path):
    data = path.read_bytes()
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File")
    header_length = struct.unpack(">I", data[4:8])[0]
    if header_length < 6 or len(data) < 8 + header_length:
        raise ValueError("Invalid MIDI header")
    midi_format, track_count, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE MIDI time division is not supported")
    if division == 0:
        raise ValueError("MIDI ticks per beat cannot be zero")
    offset = 8 + header_length
    note_events = []
    tempo_events = []
    lyrics = []
    for track_number in range(track_count):
        if offset + 8 > len(data) or data[offset : offset + 4] != b"MTrk":
            raise ValueError(f"Missing MIDI track {track_number}")
        length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        start = offset + 8
        end = start + length
        if end > len(data):
            raise ValueError(f"Truncated MIDI track {track_number}")
        events, tempos, track_lyrics = _parse_track(data[start:end], track_number)
        note_events.extend(events)
        tempo_events.extend(tempos)
        lyrics.extend(track_lyrics)
        offset = end

    tempo_events.sort(key=lambda event: (event[0], event[2], event[3]))
    collapsed_tempos = []
    for tick, microseconds, *_ in tempo_events:
        if collapsed_tempos and collapsed_tempos[-1][0] == tick:
            collapsed_tempos[-1] = (tick, microseconds)
        else:
            collapsed_tempos.append((tick, microseconds))
    if not collapsed_tempos or collapsed_tempos[0][0] != 0:
        collapsed_tempos.insert(0, (0, 500_000))

    def tick_seconds(target_tick: int) -> float:
        seconds = 0.0
        previous_tick = 0
        tempo = 500_000
        for change_tick, next_tempo in collapsed_tempos:
            if change_tick > target_tick:
                break
            seconds += (change_tick - previous_tick) * tempo / division / 1_000_000
            previous_tick = change_tick
            tempo = next_tempo
        return seconds + (target_tick - previous_tick) * tempo / division / 1_000_000

    active = defaultdict(list)
    notes = []
    for tick, track, order, kind, channel, pitch, velocity in sorted(note_events):
        key = (track, channel, pitch)
        if kind == "on":
            active[key].append((tick, velocity))
        elif active[key]:
            start_tick, start_velocity = active[key].pop(0)
            if tick > start_tick:
                notes.append(
                    {
                        "pitch": pitch,
                        "channel": channel,
                        "velocity": start_velocity,
                        "start_tick": start_tick,
                        "duration_ticks": tick - start_tick,
                        "start_seconds": tick_seconds(start_tick),
                        "duration_seconds": tick_seconds(tick) - tick_seconds(start_tick),
                    }
                )
    notes.sort(key=lambda note: (note["start_tick"], note["pitch"]))
    rests = []
    occupied_until = 0
    for note in notes:
        if note["start_tick"] > occupied_until:
            rests.append(
                {
                    "start_tick": occupied_until,
                    "duration_ticks": note["start_tick"] - occupied_until,
                    "start_seconds": tick_seconds(occupied_until),
                    "duration_seconds": tick_seconds(note["start_tick"]) - tick_seconds(occupied_until),
                }
            )
        occupied_until = max(occupied_until, note["start_tick"] + note["duration_ticks"])
    tempos = [
        {"tick": tick, "bpm": 60_000_000 / microseconds, "seconds": tick_seconds(tick)}
        for tick, microseconds in collapsed_tempos
    ]
    return {
        "format": midi_format,
        "tracks": track_count,
        "ticks_per_beat": division,
        "tempos": tempos,
        "notes": notes,
        "rests": rests,
        "lyrics": sorted(lyrics, key=lambda lyric: lyric["tick"]),
        "unclosed_notes": sum(len(items) for items in active.values()),
    }
