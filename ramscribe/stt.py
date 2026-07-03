"""Local speech-to-text over a sliding window, numpy-in only.

`faster-whisper` accepts a numpy array of samples directly, so audio is handed
to the model straight from the ring buffer. No temp WAV, no file of any kind is
ever created here. The only thing that leaves this module is text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    t_start: float
    t_end: float
    text: str
    speaker: str = "me"  # channel-derived; "me" (mic) in MVP, "them" for system audio

    def as_dict(self) -> dict:
        return {
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "text": self.text.strip(),
            "speaker": self.speaker,
        }


class WhisperAdapter:
    """Thin wrapper around faster-whisper's WhisperModel.

    Kept deliberately small so the sliding-window logic can be tested against a
    stub with the same `.transcribe()` shape (no model download required).
    """

    def __init__(self, model_name: str = "small", device: str = "cpu",
                 compute_type: str = "int8", language: str | None = None):
        from faster_whisper import WhisperModel  # local import: heavy dependency

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio, sample_rate: int = 16000) -> list[tuple[float, float, str]]:
        # audio is an in-memory float32 numpy array. Never persisted.
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return [(s.start, s.end, s.text) for s in segments]


class StubTranscriber:
    """Deterministic fake used by the audit probe and tests.

    Emits one short "segment" per second of audio so the full pipeline can run
    headless without downloading a model. Produces text only.
    """

    def __init__(self, language: str | None = None):
        self.language = language

    def transcribe(self, audio, sample_rate: int = 16000) -> list[tuple[float, float, str]]:
        import numpy as np

        n = len(audio)
        out: list[tuple[float, float, str]] = []
        secs = int(n // sample_rate)
        for i in range(secs):
            chunk = audio[i * sample_rate:(i + 1) * sample_rate]
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
            out.append((float(i), float(i + 1), f"[synthetic audio rms={rms:.3f}]"))
        return out


class SlidingWindowTranscriber:
    """Turns overlapping window transcriptions into a stream of final segments.

    Each call to `process` transcribes the tail of the current buffer. Segments
    that end before the provisional margin become final exactly once; newer
    segments are returned as provisional text that may still change.
    """

    def __init__(self, model, sample_rate: int = 16000, overlap_s: float = 5.0,
                 speaker: str = "me"):
        self._model = model
        self.sample_rate = sample_rate
        self.overlap_s = overlap_s
        # Speaker is a property of the *channel*, not the voice: every segment
        # this instance emits is stamped with it. The mic pipeline is "me"; a
        # second (system-audio) pipeline would be "them". No voice analysis.
        self.speaker = speaker
        self.finalized_until = 0.0  # absolute seconds already emitted as final

    def process(self, audio, window_start: float, window_end: float,
                final: bool = False) -> tuple[list[Segment], str]:
        """Transcribe a window; return (newly_final_segments, provisional_text).

        `audio` covers [window_start, window_end] in absolute stream seconds.
        On `final=True` (shutdown) everything left is emitted as final.
        """
        if len(audio) == 0:
            return [], ""

        # Only feed the model from where we still need transcription, minus a bit
        # of overlap for context. Keeps each pass cheap and stable.
        t0 = max(window_start, self.finalized_until - self.overlap_s)
        offset = int(round((t0 - window_start) * self.sample_rate))
        offset = max(0, min(offset, len(audio)))
        clip = audio[offset:]
        if len(clip) == 0:
            return [], ""

        raw = self._model.transcribe(clip, sample_rate=self.sample_rate)

        provisional_cutoff = window_end if final else (window_end - self.overlap_s)
        new_final: list[Segment] = []
        provisional_parts: list[str] = []
        max_final_end = self.finalized_until

        for rel_start, rel_end, text in raw:
            abs_start = t0 + rel_start
            abs_end = t0 + rel_end
            text = text.strip()
            if not text:
                continue
            # Already emitted in an earlier pass.
            if abs_end <= self.finalized_until + 1e-6:
                continue
            if abs_end <= provisional_cutoff:
                new_final.append(Segment(abs_start, abs_end, text, speaker=self.speaker))
                max_final_end = max(max_final_end, abs_end)
            else:
                provisional_parts.append(text)

        if max_final_end > self.finalized_until:
            self.finalized_until = max_final_end

        return new_final, " ".join(provisional_parts).strip()
