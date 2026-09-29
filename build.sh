#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Virtual environment created at .venv. Activate it manually on this platform." >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall -q .
pytest -q

echo "graphRagX environment is ready. Start with: python main.py"
