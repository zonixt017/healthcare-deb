#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-7860}"
ADDRESS="${STREAMLIT_ADDRESS:-0.0.0.0}"

exec streamlit run app.py \
  --server.port="${PORT}" \
  --server.address="${ADDRESS}" \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=true
