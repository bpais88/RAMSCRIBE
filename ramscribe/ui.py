"""Terminal UI: a live transcript panel plus the persistence-boundary status bar.

The status bar is the demo's whole argument. It shows, live: how much audio is
in RAM right now, the age of the oldest sample, how many segments were
finalized, and the punchline that never changes: audio bytes written to disk: 0.
"""

from __future__ import annotations

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TranscriptView:
    def __init__(self, max_lines: int = 200):
        self.max_lines = max_lines
        self._final_lines: list[str] = []
        self._provisional: str = ""
        self._finalized_count = 0
        self._buffer_stats = {"fill_seconds": 0.0, "max_seconds": 30.0, "oldest_age_seconds": 0.0}
        self._rms = 0.0
        self._transcript_path = ""

    def add_final(self, segment) -> None:
        ts = f"[{segment.t_start:6.1f}s]"
        self._final_lines.append(f"{ts} {segment.text}")
        if len(self._final_lines) > self.max_lines:
            self._final_lines = self._final_lines[-self.max_lines:]
        self._finalized_count += 1

    def set_provisional(self, text: str) -> None:
        self._provisional = text

    def update_stats(self, buffer_stats: dict, rms: float, transcript_path: str) -> None:
        self._buffer_stats = buffer_stats
        self._rms = rms
        self._transcript_path = transcript_path

    # --- rendering ---------------------------------------------------------

    def _transcript_panel(self) -> Panel:
        body = Text()
        tail = self._final_lines[-40:]
        if not tail and not self._provisional:
            body.append("listening…\n", style="dim")
        for line in tail:
            body.append(line + "\n")
        if self._provisional:
            body.append(self._provisional, style="dim italic")
            body.append("  ▌\n", style="dim")
        return Panel(body, title="RamScribe — live transcript", border_style="cyan")

    def _status_panel(self) -> Panel:
        s = self._buffer_stats
        fill = s.get("fill_seconds", 0.0)
        mx = s.get("max_seconds", 30.0)
        age = s.get("oldest_age_seconds", 0.0)
        ratio = min(fill / mx, 1.0) if mx else 0.0
        bar_len = 24
        filled = int(round(ratio * bar_len))
        bar = "█" * filled + "░" * (bar_len - filled)

        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")

        table.add_row(
            Text.assemble(("audio in RAM: ", "bold"),
                          (f"{fill:5.1f}s / {mx:.0f}s max  ", "green"),
                          (bar, "green")),
            Text.assemble(("mic RMS: ", "dim"), (f"{self._rms:.3f}", "cyan")),
        )
        table.add_row(
            Text.assemble(("oldest audio sample age: ", "bold"),
                          (f"{age:5.1f}s", "yellow")),
            Text.assemble(("segments finalized: ", "bold"),
                          (f"{self._finalized_count}", "cyan")),
        )
        table.add_row(
            Text.assemble(("audio bytes written to disk: ", "bold"),
                          ("0 (by design)", "bold green")),
            Text(self._transcript_path, style="dim"),
        )
        return Panel(table, title="persistence boundary", border_style="green")

    def render(self) -> Group:
        layout = Layout()
        layout.split_column(
            Layout(self._transcript_panel(), name="transcript"),
            Layout(self._status_panel(), name="status", size=5),
        )
        return Group(layout)
