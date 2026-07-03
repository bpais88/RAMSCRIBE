.PHONY: run audit test install list-devices clean

# Default to python3 so `make` works on macOS (no bare `python`). Inside an
# activated venv this still resolves to the venv interpreter. Override with
# e.g. `make run PY=python3.11` if needed.
PY ?= python3

install:
	$(PY) -m pip install -e ".[dev]"

run:
	$(PY) -m ramscribe

list-devices:
	$(PY) -m ramscribe --list-devices

audit:
	$(PY) scripts/audit_boundary.py

test:
	$(PY) -m pytest -q

clean:
	rm -f transcripts/session-*.jsonl
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
