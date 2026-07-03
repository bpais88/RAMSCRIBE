"""Audio capture: microphone (or in-memory synthetic source) -> ring buffer.

The sounddevice callback receives float32 PCM frames and pushes them straight
into the RAM ring buffer. Nothing is buffered to disk; we only ever log counts,
durations, and RMS levels — never the samples themselves.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from .ring import RingBuffer


class MicCapture:
    """Capture from the default (or chosen) input device at 16 kHz mono."""

    def __init__(self, ring: RingBuffer, device: int | None = None,
                 blocksize: int = 1600):
        self.ring = ring
        self.device = device
        self.blocksize = blocksize
        self.sample_rate = ring.sample_rate
        self._stream = None
        self._last_rms = 0.0

    def _callback(self, indata, frames, time_info, status):
        # indata: float32 (frames, channels). Downmix to mono, hand to RAM ring.
        mono = indata[:, 0] if indata.ndim > 1 else indata
        self._last_rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
        self.ring.write(mono)

    def start(self) -> None:
        import sounddevice as sd  # local import: needs PortAudio present

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def last_rms(self) -> float:
        return self._last_rms


class SyntheticCapture:
    """Headless source: generates a sine tone in memory and feeds the ring.

    Used by the boundary audit and tests so the real pipeline can run without a
    microphone or PortAudio. No audio is read from or written to disk.
    """

    def __init__(self, ring: RingBuffer, freq: float = 220.0, blocksize: int = 1600):
        self.ring = ring
        self.freq = freq
        self.blocksize = blocksize
        self.sample_rate = ring.sample_rate
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._phase = 0
        self._last_rms = 0.0

    def _run(self) -> None:
        period = self.blocksize / self.sample_rate
        while not self._stop.is_set():
            t = (np.arange(self.blocksize) + self._phase) / self.sample_rate
            block = (0.2 * np.sin(2 * math.pi * self.freq * t)).astype(np.float32)
            self._phase += self.blocksize
            self._last_rms = float(np.sqrt(np.mean(np.square(block))))
            self.ring.write(block)
            time.sleep(period)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def last_rms(self) -> float:
        return self._last_rms
