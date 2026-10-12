#!/usr/bin/env bash
# Run the smoke test without Docker (Linux/macOS, Python 3.12).
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
. .venv/bin/activate
pip install --quiet pip==25.2
pip install --quiet --no-deps -r requirements.txt
pip check
python -m pytest -q
python -m src.benchmark --config "${1:-configs/smoke.yaml}"
