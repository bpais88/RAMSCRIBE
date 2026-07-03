"""Shared constants and tiny helpers — the values that *are* the contract.

`SAMPLE_RATE`, `MAX_SECONDS`, and especially `AUDIO_EXTS` are consumed by the
pipeline, the boundary audit, and the persistence test alike. They live here so
there is exactly one source of truth: the banned-extension list drifting between
the auditor and the test would be a hole in the guarantee this repo exists to
make.
"""

from __future__ import annotations

import math

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0

# Audio-typed file extensions that must never be written anywhere in the repo.
# The auditor greps for these and the test asserts none appear on disk.
AUDIO_EXTS = frozenset({
    ".wav", ".mp3", ".ogg", ".flac", ".pcm", ".npy", ".npz",
    ".m4a", ".aac", ".wma", ".aiff",
})


def rms(samples) -> float:
    """Root-mean-square level of a sample block, allocation-free.

    Uses a dot product rather than `sqrt(mean(square(x)))` so no temporary array
    is created — this runs on the audio callback thread where alloc jitter is
    best avoided. Returns 0.0 for an empty block.
    """
    n = len(samples)
    if n == 0:
        return 0.0
    return math.sqrt(float(samples @ samples) / n)
