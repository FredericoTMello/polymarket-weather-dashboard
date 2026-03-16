#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BIND="${1:-127.0.0.1}"
PORT="${2:-8090}"

while true; do
    echo "[$(date)] Starting server..."
    python3 server.py "$BIND" "$PORT"
    echo "[$(date)] Server died, restarting in 3s..."
    sleep 3
done
