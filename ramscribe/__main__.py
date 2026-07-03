"""RamScribe CLI: RAM-only live transcriber.

    mic ─callback→ ring buffer (RAM, ≤30s) ─every ~3s→ faster-whisper(numpy)
        → finalized text → JSONL + terminal

Audio lives only in the ring buffer for a few seconds. The transcript is the
only artifact. Run with `python -m ramscribe`.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime

from .capture import MicCapture, SyntheticCapture
from .ring import RingBuffer
from .sink import TranscriptSink
from .stt import SlidingWindowTranscriber, StubTranscriber
from .ui import TranscriptView

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0
TRANSCRIBE_EVERY = 3.0
OVERLAP_SECONDS = 5.0


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def list_devices() -> None:
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on PortAudio
        print(f"Could not query devices: {exc}", file=sys.stderr)
        return
    print(sd.query_devices())


def build_model(args):
    if args.stub_stt:
        return StubTranscriber(language=args.lang)
    from .stt import WhisperAdapter

    return WhisperAdapter(
        model_name=args.model,
        device="cpu",
        compute_type="int8",
        language=args.lang,
    )


def run(args) -> int:
    ring = RingBuffer(sample_rate=SAMPLE_RATE, max_seconds=MAX_SECONDS)
    sink = TranscriptSink(timestamp=_timestamp())

    if args.source == "synthetic":
        capture = SyntheticCapture(ring)
    else:
        capture = MicCapture(ring, device=args.device)

    model = build_model(args)
    transcriber = SlidingWindowTranscriber(model, sample_rate=SAMPLE_RATE,
                                           overlap_s=OVERLAP_SECONDS)
    view = TranscriptView()

    stop = threading.Event()
    started_at = [0.0]

    def transcription_loop():
        # Wait until we have a little audio before the first pass.
        while not stop.is_set():
            audio, w_start, w_end = ring.snapshot()
            if w_end - w_start >= 1.0:
                final_segs, provisional = transcriber.process(audio, w_start, w_end)
                for seg in final_segs:
                    sink.write_segment(seg)
                    view.add_final(seg)
                view.set_provisional(provisional)
            # Sleep in small slices so shutdown is responsive.
            for _ in range(int(TRANSCRIBE_EVERY * 10)):
                if stop.is_set():
                    return
                time.sleep(0.1)

    def final_flush():
        audio, w_start, w_end = ring.snapshot()
        if w_end - w_start > 0:
            final_segs, _ = transcriber.process(audio, w_start, w_end, final=True)
            for seg in final_segs:
                sink.write_segment(seg)
                view.add_final(seg)
            view.set_provisional("")

    capture.start()
    worker = threading.Thread(target=transcription_loop, daemon=True)
    worker.start()

    started_at[0] = time.monotonic()
    deadline = started_at[0] + args.duration if args.duration else None
    exit_reason = "stopped"

    try:
        if args.no_ui:
            while not stop.is_set():
                if deadline and time.monotonic() >= deadline:
                    break
                stats = ring.stats()
                if not args.quiet:
                    print(
                        f"audio in RAM: {stats['fill_seconds']:.1f}s / "
                        f"{stats['max_seconds']:.0f}s | oldest {stats['oldest_age_seconds']:.1f}s "
                        f"| finalized {sink.count} | audio bytes to disk: 0 (by design)",
                        flush=True,
                    )
                time.sleep(0.5)
        else:
            from rich.live import Live

            with Live(view.render(), refresh_per_second=8, screen=False) as live:
                while not stop.is_set():
                    if deadline and time.monotonic() >= deadline:
                        break
                    view.update_stats(ring.stats(), capture.last_rms, str(sink.path))
                    live.update(view.render())
                    time.sleep(0.1)
    except KeyboardInterrupt:
        exit_reason = "Ctrl+C"
    finally:
        stop.set()
        worker.join(timeout=TRANSCRIBE_EVERY + 1.0)
        capture.stop()
        # One last pass so trailing speech is not lost, then wipe the RAM buffer.
        try:
            final_flush()
        except Exception as exc:  # pragma: no cover - best-effort on shutdown
            print(f"(final flush skipped: {exc})", file=sys.stderr)
        ring.clear()

    print()
    print(f"RamScribe {exit_reason}. Transcript: {sink.path}")
    print(f"Finalized segments: {sink.count}. Audio bytes written to disk: 0 (by design).")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ramscribe",
        description="RAM-only live transcriber. Audio never touches disk.",
    )
    p.add_argument("--lang", default=None,
                   help="Force language (e.g. en, de, pt). Default: auto-detect.")
    p.add_argument("--model", default="small",
                   help="faster-whisper model size (tiny/base/small/...). Default: small.")
    p.add_argument("--device", type=int, default=None,
                   help="Input device index (see --list-devices).")
    p.add_argument("--list-devices", action="store_true",
                   help="List available audio input devices and exit.")
    p.add_argument("--source", choices=["mic", "synthetic"], default="mic",
                   help="Audio source. 'synthetic' feeds an in-memory tone (headless).")
    p.add_argument("--stub-stt", action="store_true",
                   help="Use the deterministic stub transcriber (no model download).")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Auto-stop after N seconds (0 = run until Ctrl+C).")
    p.add_argument("--no-ui", action="store_true",
                   help="Headless: plain stdout status instead of the rich UI.")
    p.add_argument("--quiet", action="store_true",
                   help="With --no-ui, suppress the periodic status line.")
    args = p.parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
