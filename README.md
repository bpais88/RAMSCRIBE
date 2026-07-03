# RamScribe

**A RAM-only live transcriber.** RamScribe listens to a live conversation and
produces a **text transcript only**. Audio exists exclusively in volatile
memory (a bounded ring buffer) for a few seconds and is **never written to disk
in any form**. The transcript is the artifact; the audio is transit data.

This is a local proof-of-concept demonstrating the "RAM-only, transcript-first"
compliance architecture — not a product.

```
mic ─sounddevice callback→ ring buffer (RAM, ≤30s) ─every ~3s→
   faster-whisper.transcribe(numpy array in memory) → finalized text → JSONL + terminal
```

## Why RAM-only

In Germany, recording a confidential spoken word without consent can violate
**§201 StGB** (Verletzung der Vertraulichkeit des Wortes) — the *recording*
itself is the offense. Under **GDPR**, voice is personal (often special-category)
data, and the lawful basis here is **legitimate interest (Art. 6(1)(f))**: you
need the *information* from a meeting, not a recording of people's voices. By
keeping audio only in a bounded RAM buffer and persisting **only the transcript**,
the data-minimisation and storage-limitation principles are satisfied by
construction. **This demo produces no audio artifact and cannot replay
anything** — the moment a sample is transcribed or ages out of the 30s window,
it is gone.

## The persistence boundary

These are enforced in code and verified automatically (`make audit`, `make test`):

- **No audio ever touches disk.** No `soundfile.write`, `wavfile.write`,
  `wave.open(...,'wb')`, `np.save`, `tofile`, pickling of buffers, or `tempfile`
  in the audio path. No `.wav/.mp3/.ogg/.flac/.pcm/.npy` audio is ever written.
- **No hidden persistence.** No STT/capture caching options that dump audio; logs
  contain only counts, durations, and RMS levels — never raw samples.
- **Bounded buffer.** The ring buffer is a preallocated numpy array that
  *structurally* cannot exceed 30 seconds, regardless of load.
- The status bar's punchline stays on screen the whole run:
  `audio bytes written to disk: 0 (by design)`.

## Install

Requires Python 3.11+ and PortAudio (for microphone capture).

```bash
# with uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# or plain pip (use python3 on macOS — there is no bare `python`)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Once the venv is activated, bare `python`/`pip` and `make run` all resolve to
the venv interpreter. Outside a venv on macOS, use `python3`.

On first run, `faster-whisper` downloads the `small` model weights (~460 MB) to
its cache. Those are **model weights, not audio** — audio never lands on disk.

> PortAudio: on Debian/Ubuntu `sudo apt-get install libportaudio2`;
> on macOS `brew install portaudio`.

## Run

```bash
make run                      # live transcription from the default mic
python -m ramscribe --lang en # force a language (en / de / pt / ...)
python -m ramscribe --model base   # smaller/faster model
make list-devices             # show input devices and their indices
```

Speak into the mic — your words appear as live text within ~3–5s. Provisional
text renders dim/italic at the bottom and firms up into final segments. Press
**Ctrl+C** for a clean shutdown: a final transcription pass runs, the RAM buffer
is zeroed, and the transcript path is printed.

Finalized segments are appended to `transcripts/session-<timestamp>.jsonl`, one
JSON object per segment:

```json
{"t_start": 3.12, "t_end": 5.44, "text": "hello, this is a test", "speaker": "me"}
```

`speaker` is always `"me"` in the mic-only MVP. The field exists from day one so
the dual-channel stretch (below) slots in with no schema change — and it is
derived from the audio *channel*, never from analysing the voice.

## Verify the boundary

```bash
make audit   # (a) greps source for banned audio-persistence APIs
             # (b) runs a real 10s session while watching the repo + $TMPDIR
             #     for any new audio-typed file — fails if one appears
make test    # monkeypatches every audio-write path to raise, then runs the
             # pipeline over 5s of in-memory synthetic audio; asserts nothing
             # audio-like is ever written
```

The audit's dynamic probe runs the *actual* pipeline (`--source synthetic
--stub-stt`) so it needs no microphone or model download while still exercising
every code path that could conceivably persist audio.

## Flags

| Flag | Meaning |
|------|---------|
| `--lang en/de/pt` | Force language (default: auto-detect) |
| `--model small` | faster-whisper model size (`tiny`/`base`/`small`/…) |
| `--device N` | Input device index (see `--list-devices`) |
| `--list-devices` | List input devices and exit |
| `--source synthetic` | Feed an in-memory tone instead of the mic (headless) |
| `--stub-stt` | Deterministic stub transcriber (no model download) |
| `--duration N` | Auto-stop after N seconds (0 = until Ctrl+C) |
| `--no-ui` | Plain stdout status instead of the rich UI |

## Stretch (not built in the MVP): dual-channel VoIP capture

This is the upgrade that turns the demo into the full "Bliro pattern" —
transcribe a real VoIP/Zoom call with **speaker attribution from channel
metadata, not voice biometrics**. It is planned, not yet implemented.

The design: run two `sounddevice` streams in parallel — the microphone (you) and
a loopback device carrying system audio (all remote parties). Each stream feeds
its **own** independent ring buffer + sliding-window transcriber (two instances
of the existing pipeline — channels are never mixed into one buffer). Mic
segments are stamped `speaker: "me"`, system segments `speaker: "them"`. The UI
merges both streams by timestamp into one interleaved transcript with coloured
`me:` / `them:` prefixes, and the status bar shows buffer stats for both
channels. `make audit` and the persistence test must pass unchanged with both
streams live.

Planned flags: `--device-mic <name>` and `--device-system <name>`. On macOS the
loopback is [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole)
(`brew install blackhole-2ch`), paired with a Multi-Output Device in Audio MIDI
Setup so you still hear the call.

**Why this matters.** Speaker identity comes from *which channel the audio
arrived on*, never from analysing the voice itself — which keeps biometric
(GDPR Art. 9) processing entirely out of the system. Honest platform boundary:
this OS-level tap works for anything playing through the desktop (Zoom, Teams,
any softphone), but mobile OSes prohibit apps from tapping phone/VoIP call
audio — capturing mobile calls requires a server-side telephony approach (e.g.
Twilio Media Streams) feeding the same RAM-only pipeline.

## Non-goals

No system-audio/loopback capture in the MVP itself (it is the defined stretch
above), no meeting-bot, and **no voice-based diarization of any kind — ever**,
including in the stretch: speaker labels come exclusively from channel metadata.
No cloud APIs, no accounts, no packaging/installer, no GUI beyond the terminal.
