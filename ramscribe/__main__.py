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
from .config import MAX_SECONDS, SAMPLE_RATE
from .ring import RingBuffer
from .sink import TranscriptSink
from .stt import SlidingWindowTranscriber, StubTranscriber
from .ui import TranscriptView

TRANSCRIBE_EVERY = 3.0  # seconds between transcription passes
OVERLAP_SECONDS = 5.0   # sliding-window context overlap


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

    if not args.stub_stt:
        # Loading (and, on first run, downloading) the model is a silent pause of
        # several seconds before the UI takes over — say so, or it reads as a hang.
        print(f"Loading '{args.model}' model… "
              f"(first run downloads weights, which can take a minute; "
              f"then the live view opens — don't Ctrl+C)", flush=True)
    model = build_model(args)
    transcriber = SlidingWindowTranscriber(model, sample_rate=SAMPLE_RATE,
                                           overlap_s=OVERLAP_SECONDS)
    view = TranscriptView()

    stop = threading.Event()

    def emit(final_segs, provisional):
        # The one place a finished transcription pass turns into output.
        for seg in final_segs:
            sink.write_segment(seg)
            view.add_final(seg)
        view.set_provisional(provisional)

    def transcription_loop():
        # Wait until we have a little audio before the first pass.
        while not stop.is_set():
            audio, w_start, w_end = ring.snapshot()
            if w_end - w_start >= 1.0:
                emit(*transcriber.process(audio, w_start, w_end))
            # Sleep in small slices so shutdown is responsive.
            for _ in range(int(TRANSCRIBE_EVERY * 10)):
                if stop.is_set():
                    return
                time.sleep(0.1)

    def final_flush():
        audio, w_start, w_end = ring.snapshot()
        if w_end - w_start > 0:
            final_segs, _ = transcriber.process(audio, w_start, w_end, final=True)
            emit(final_segs, "")

    capture.start()
    worker = threading.Thread(target=transcription_loop, daemon=True)
    worker.start()

    deadline = time.monotonic() + args.duration if args.duration else None
    exit_reason = "stopped"

    def drive(on_tick, interval):
        # Single stop/deadline/tick driver shared by the headless and rich loops.
        while not stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            on_tick()
            time.sleep(interval)

    try:
        if args.no_ui:
            def tick():
                if args.quiet:
                    return
                s = ring.stats()
                print(
                    f"audio in RAM: {s['fill_seconds']:.1f}s / {s['max_seconds']:.0f}s "
                    f"| oldest {s['oldest_age_seconds']:.1f}s | finalized {sink.count} "
                    f"| audio bytes to disk: 0 (by design)",
                    flush=True,
                )
            drive(tick, 0.5)
        else:
            from rich.live import Live

            with Live(view.render(), refresh_per_second=8, screen=False) as live:
                def tick():
                    view.update_stats(ring.stats(), capture.last_rms, str(sink.path))
                    live.update(view.render())
                drive(tick, 0.1)
    except KeyboardInterrupt:
        exit_reason = "Ctrl+C"
    finally:
        stop.set()
        # Wait for the worker to fully exit before the final pass. Once `stop` is
        # set it returns right after its current transcribe, so this join is
        # bounded in practice — and it guarantees final_flush() has sole access
        # to the transcriber and sink, so segments can't race or duplicate.
        worker.join()
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
