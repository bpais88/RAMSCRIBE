"""Text sink: append finalized transcript segments to a JSONL file.

This is the only thing RamScribe ever persists — one JSON object per finalized
segment. It is plain UTF-8 text opened in append mode; there is no path here
through which audio bytes could be written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class TranscriptSink:
    def __init__(self, transcripts_dir: str | Path = "transcripts", timestamp: str | None = None):
        self.dir = Path(transcripts_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = timestamp or "session"
        self.path = self.dir / f"session-{stamp}.jsonl"
        self._count = 0
        # Touch the file so the path exists even for a silent session.
        self.path.touch(exist_ok=True)

    def write_segment(self, segment) -> None:
        """Append one finalized segment as a JSON line (text only)."""
        obj = segment.as_dict() if hasattr(segment, "as_dict") else dict(segment)
        line = json.dumps(obj, ensure_ascii=False)
        # Text mode, append: no binary, no audio.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._count += 1

    @property
    def count(self) -> int:
        return self._count
