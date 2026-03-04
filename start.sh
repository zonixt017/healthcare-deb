#!/usr/bin/env bash
set -euo pipefail

export PORT="${PORT:-7860}"
export APP_HOST="${APP_HOST:-0.0.0.0}"

exec python app.py
