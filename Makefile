.PHONY: run audit test install list-devices clean

PY ?= python

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
