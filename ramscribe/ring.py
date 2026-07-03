"""Bounded, in-memory ring buffer for live audio.

The whole point of RamScribe lives here: audio only ever exists as a fixed-size
window of float32 samples in RAM. The buffer is preallocated and *cannot*
structurally grow past its capacity, no matter how much audio is pushed through
it. Old samples are overwritten in place; nothing is spooled, cached, or
written anywhere.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """A thread-safe, fixed-capacity ring of float32 audio samples.

    Capacity is `sample_rate * max_seconds` samples, allocated once. Writes wrap
    around; the buffer never holds more than `max_seconds` of audio.
    """

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 30.0):
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.capacity = int(round(sample_rate * max_seconds))
        # Preallocated. This is the only place audio samples ever live.
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0          # index of next slot to write
        self._filled = 0             # valid samples currently in buffer (<= capacity)
        self._total_written = 0      # monotonic count of every sample ever accepted
        self._lock = threading.Lock()

    def write(self, frames: np.ndarray) -> None:
        """Append mono float32 samples, overwriting the oldest as needed."""
        frames = np.asarray(frames, dtype=np.float32).reshape(-1)
        n = frames.size
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
                # More samples than the whole buffer: keep only the newest tail.
                self._buf[:] = frames[-self.capacity:]
                self._write_pos = 0
                self._filled = self.capacity
                self._total_written += n
                return
            end = self._write_pos + n
            if end <= self.capacity:
                self._buf[self._write_pos:end] = frames
            else:
                first = self.capacity - self._write_pos
                self._buf[self._write_pos:] = frames[:first]
                self._buf[: n - first] = frames[first:]
            self._write_pos = end % self.capacity
            self._filled = min(self._filled + n, self.capacity)
            self._total_written += n

    def snapshot(self) -> tuple[np.ndarray, float, float]:
        """Return (audio_copy, start_time, end_time) for the current window.

        Times are absolute seconds measured from the start of the stream, so the
        transcriber can place segments on a stable timeline even as the buffer
        wraps. `audio_copy` is a fresh contiguous array in chronological order.
        """
        with self._lock:
            filled = self._filled
            total = self._total_written
            if filled == 0:
                empty = np.zeros(0, dtype=np.float32)
                t = total / self.sample_rate
                return empty, t, t
            start_idx = (self._write_pos - filled) % self.capacity
            if start_idx + filled <= self.capacity:
                audio = self._buf[start_idx:start_idx + filled].copy()
            else:
                first = self.capacity - start_idx
                audio = np.empty(filled, dtype=np.float32)
                audio[:first] = self._buf[start_idx:]
                audio[first:] = self._buf[: filled - first]
        end_time = total / self.sample_rate
        start_time = (total - filled) / self.sample_rate
        return audio, start_time, end_time

    def stats(self) -> dict:
        """Cheap counters for the status bar. Never returns sample data."""
        with self._lock:
            filled = self._filled
        seconds = filled / self.sample_rate
        return {
            "fill_seconds": seconds,
            "max_seconds": self.max_seconds,
            "oldest_age_seconds": seconds,  # oldest live sample = age of the window
            "samples": filled,
            "capacity_seconds": self.capacity / self.sample_rate,
        }

    def clear(self) -> None:
        """Zero the backing memory. Called on shutdown so no audio lingers."""
        with self._lock:
            self._buf.fill(0)
            self._write_pos = 0
            self._filled = 0
