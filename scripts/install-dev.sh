#!/usr/bin/env bash
set -euo pipefail

# Installs dev requirements and optional type stubs into the active environment.
# Usage: ./scripts/install-dev.sh

## Determine which Python to use for installing packages.
# Priority:
# 1. Active virtualenv (VIRTUAL_ENV)
# 2. Local project venv at `./.venv` or `./venv` if present
# 3. Fallback to system `python3`
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [ -x "./.venv/bin/python" ]; then
  PYTHON="./.venv/bin/python"
elif [ -x "./venv/bin/python" ]; then
  PYTHON="./venv/bin/python"
else
  PYTHON="python3"
fi

# Use `python -m pip` for robustness (works whether inside venv or not).
PIP="$PYTHON -m pip"

echo "Using Python: $PYTHON"

echo "Installing runtime requirements..."
$PIP install -r requirements.txt

echo "Installing dev requirements (linters, mypy, etc.)..."
$PIP install -r requirements-dev.txt

echo "Attempting to install type stubs from types-requirements.txt (optional)..."
# Try to install type stubs but don't fail the whole script if some stubs
# are not available on PyPI for this platform.
if [ -f types-requirements.txt ]; then
  if $PIP install -r types-requirements.txt; then
    echo "Type stubs installed."
  else
    echo "Warning: some type-stub packages failed to install."
    echo "You can edit types-requirements.txt to remove unavailable packages"
    echo "or install stubs manually. Continuing without failing."
  fi
else
  echo "No types-requirements.txt found; skipping type stubs install."
fi

echo "Done."

