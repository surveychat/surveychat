#!/usr/bin/env bash
set -euo pipefail

# Convenience script for local testing or simple servers.
# Override these with environment variables if needed.
SERVER_NAME="${SERVER_NAME:-localhost}"
PORT="${PORT:-8501}"

streamlit run app.py \
  --browser.serverAddress "$SERVER_NAME" \
  --server.port "$PORT"
