#!/usr/bin/env python3
"""Persistence-boundary audit for RamScribe.

Two independent checks, either failure exits non-zero:

  (a) Static: grep the source tree for banned audio-persistence APIs and file
      extensions. Any hit fails the audit.

  (b) Dynamic: run a real ~10s session in a subprocess while taking a
      before/after inventory of every file under the repo and $TMPDIR. If any
      new audio-typed file appears anywhere, the audit fails.

Run via `make audit`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from ramscribe.config import AUDIO_EXTS  # noqa: E402 — single source for the banned-extension list

# We scan the audio pipeline package. tests/ and scripts/audit_boundary.py are
# the *enforcement* layer — they name the banned APIs on purpose (to block them),
# so they are intentionally out of scope for the ban.
SOURCE_DIRS = ["ramscribe"]

# Banned APIs / patterns. Each entry: (compiled regex, human description).
BANNED_PATTERNS = [
    (r"soundfile\s*\.\s*write", "soundfile.write (audio -> disk)"),
    (r"scipy\.io\.wavfile\.write", "scipy.io.wavfile.write (audio -> disk)"),
    (r"wavfile\s*\.\s*write", "wavfile.write (audio -> disk)"),
    (r"wave\s*\.\s*open\s*\([^)]*['\"][wa]b?['\"]", "wave.open(..., 'wb') (audio -> disk)"),
    (r"np(?:umpy)?\s*\.\s*save\b", "numpy.save (buffer -> .npy)"),
    (r"np(?:umpy)?\s*\.\s*savez", "numpy.savez (buffer -> .npz)"),
    (r"\bpickle\s*\.\s*dump", "pickle.dump (buffer -> disk)"),
    (r"\btempfile\b", "tempfile usage (audio path must not touch temp files)"),
    (r"\.tofile\s*\(", "ndarray.tofile (raw samples -> disk)"),
    (r"AudioSegment\s*\.\s*export", "pydub AudioSegment.export (audio -> disk)"),
]

# Files that are *allowed* to mention the banned names (this auditor lists them).
ALLOWLIST = {str((REPO_ROOT / "scripts" / "audit_boundary.py").resolve())}


def _iter_source_files():
    for d in SOURCE_DIRS:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            yield path


def check_static() -> list[str]:
    violations: list[str] = []
    compiled = [(re.compile(p), desc) for p, desc in BANNED_PATTERNS]
    for path in _iter_source_files():
        if str(path.resolve()) in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for rx, desc in compiled:
                if rx.search(line):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{i}: {desc}  -> {stripped}")
    return violations


def _inventory(root: Path) -> dict[str, int]:
    inv: dict[str, int] = {}
    if not root.exists():
        return inv
    for p in root.rglob("*"):
        try:
            if p.is_file():
                inv[str(p.resolve())] = p.stat().st_size
        except (OSError, ValueError):
            continue
    return inv


def _audio_files(paths) -> list[str]:
    return [p for p in paths if Path(p).suffix.lower() in AUDIO_EXTS]


def check_dynamic(duration: float = 10.0) -> list[str]:
    tmp_root = Path(tempfile.gettempdir())
    watch_roots = [REPO_ROOT, tmp_root]

    before: dict[str, int] = {}
    for r in watch_roots:
        before.update(_inventory(r))

    # A real session over the actual pipeline (ring -> stt -> sink). Synthetic
    # source + stub STT so the audit runs headless without a mic or model, while
    # still exercising every code path that could conceivably persist audio.
    cmd = [
        sys.executable, "-m", "ramscribe",
        "--source", "synthetic",
        "--stub-stt",
        "--duration", str(duration),
        "--no-ui",
        "--quiet",
    ]
    print(f"[audit] running {duration:.0f}s live probe: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=duration + 60)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return [f"probe subprocess exited with code {proc.returncode}"]

    time.sleep(0.5)
    after: dict[str, int] = {}
    for r in watch_roots:
        after.update(_inventory(r))

    new_paths = [p for p in after if p not in before]
    new_audio = _audio_files(new_paths)

    violations = [f"new audio-typed file appeared: {p}" for p in sorted(new_audio)]
    return violations


def main() -> int:
    print("=" * 70)
    print("RamScribe persistence-boundary audit")
    print("=" * 70)

    print("\n[1/2] static scan for banned audio-persistence APIs...")
    static_v = check_static()
    if static_v:
        print("  FAIL — banned APIs found:")
        for v in static_v:
            print(f"    - {v}")
    else:
        print("  OK — no banned audio-persistence APIs in source.")

    print("\n[2/2] dynamic probe: 10s live session, watching repo + $TMPDIR...")
    try:
        dyn_v = check_dynamic(10.0)
    except Exception as exc:
        dyn_v = [f"dynamic probe error: {exc}"]
    if dyn_v:
        print("  FAIL — audio persistence detected during live run:")
        for v in dyn_v:
            print(f"    - {v}")
    else:
        print("  OK — no audio-typed files created anywhere during the live run.")

    ok = not static_v and not dyn_v
    print("\n" + "=" * 70)
    print("AUDIT PASSED ✅" if ok else "AUDIT FAILED ❌")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
