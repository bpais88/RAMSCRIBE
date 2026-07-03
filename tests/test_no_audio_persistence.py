"""The persistence boundary, as an executable test.

While the pipeline processes 5 seconds of *synthetic in-memory* audio (a sine
wave built with numpy — never read from a file), we monkeypatch every plausible
audio-write path to explode:

  * `builtins.open` in any write/append mode with a binary flag or an audio
    extension,
  * `soundfile.write`,
  * `scipy.io.wavfile.write`,
  * `numpy.save`.

Writing the transcript JSONL (UTF-8 text, append mode, `.jsonl`) is explicitly
allowed. If the pipeline tries to persist audio in any form, the test fails.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from ramscribe.ring import RingBuffer
from ramscribe.sink import TranscriptSink
from ramscribe.stt import SlidingWindowTranscriber, StubTranscriber

SAMPLE_RATE = 16000
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".pcm", ".npy", ".npz", ".m4a", ".aac"}


class AudioPersistenceViolation(AssertionError):
    pass


def _make_sine(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def guarded_open(monkeypatch):
    real_open = builtins.open

    def guard(file, mode="r", *args, **kwargs):
        name = str(file).lower()
        is_write = any(flag in mode for flag in ("w", "a", "x", "+"))
        is_binary = "b" in mode
        looks_audio = any(name.endswith(ext) for ext in AUDIO_EXTS)
        if is_write and (is_binary or looks_audio):
            raise AudioPersistenceViolation(
                f"blocked audio-like write: open({file!r}, {mode!r})"
            )
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard)

    # Block the dedicated audio-writing libraries too, if present.
    try:
        import soundfile  # type: ignore

        monkeypatch.setattr(soundfile, "write",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AudioPersistenceViolation("soundfile.write blocked")))
    except Exception:
        pass
    try:
        import scipy.io.wavfile as wavfile  # type: ignore

        monkeypatch.setattr(wavfile, "write",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AudioPersistenceViolation("wavfile.write blocked")))
    except Exception:
        pass

    monkeypatch.setattr(np, "save",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AudioPersistenceViolation("np.save blocked")))
    # np.ndarray.tofile is a C-level immutable attribute and cannot be patched;
    # it is covered by the static audit instead (banned pattern `.tofile(`).
    return guard


def test_pipeline_writes_no_audio(guarded_open, tmp_path):
    ring = RingBuffer(sample_rate=SAMPLE_RATE, max_seconds=30.0)
    sink = TranscriptSink(transcripts_dir=tmp_path, timestamp="test")
    transcriber = SlidingWindowTranscriber(StubTranscriber(), sample_rate=SAMPLE_RATE)

    audio = _make_sine(5.0)
    # Feed in realistic 100 ms blocks, exactly like the capture callback.
    block = SAMPLE_RATE // 10
    for i in range(0, len(audio), block):
        ring.write(audio[i:i + block])

    snap, w_start, w_end = ring.snapshot()
    assert w_end - w_start == pytest.approx(5.0, abs=0.05)

    # Streaming passes...
    finals, provisional = transcriber.process(snap, w_start, w_end)
    # ...then a final flush.
    more, _ = transcriber.process(snap, w_start, w_end, final=True)

    all_final = finals + more
    for seg in all_final:
        sink.write_segment(seg)

    assert sink.count > 0, "expected some finalized text segments"

    # The only file produced is the transcript, and it is JSONL text.
    produced = list(Path(tmp_path).rglob("*"))
    produced_files = [p for p in produced if p.is_file()]
    assert produced_files, "sink should have produced a transcript file"
    for p in produced_files:
        assert p.suffix.lower() not in AUDIO_EXTS, f"unexpected audio file: {p}"
        assert p.suffix == ".jsonl"

    # And the boundary wipe zeroes RAM without touching disk.
    ring.clear()
    snap_after, _, _ = ring.snapshot()
    assert snap_after.size == 0


def test_ring_buffer_cannot_exceed_cap():
    ring = RingBuffer(sample_rate=SAMPLE_RATE, max_seconds=30.0)
    # Push 120 seconds through a 30 second buffer.
    for _ in range(120):
        ring.write(np.zeros(SAMPLE_RATE, dtype=np.float32))
    snap, w_start, w_end = ring.snapshot()
    assert snap.size <= ring.capacity
    assert (w_end - w_start) <= 30.0 + 1e-6
