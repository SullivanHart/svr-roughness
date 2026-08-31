#!/usr/bin/env bash
# Release script for PyPI publishing (Linux / macOS / Raspberry Pi 5)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [ ! -d "dist" ]; then
    echo "Error: dist/ directory not found. Please run ./scripts/build.sh first!"
    exit 1
fi

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    if [ -n "${TWINE_API_KEY:-}" ]; then
        export TWINE_USERNAME="__token__"
        export TWINE_PASSWORD="$TWINE_API_KEY"
        echo "Loaded TWINE_API_KEY from .env"
    fi
fi

echo "=== Installing / Upgrading Twine ==="
python3 -m pip install --upgrade twine

if [ "${1:-}" == "--test" ]; then
    echo "=== Uploading to TestPyPI ==="
    python3 -m twine upload --repository testpypi dist/*
else
    echo "=== Uploading to PyPI Production ==="
    python3 -m twine upload dist/*
fi
