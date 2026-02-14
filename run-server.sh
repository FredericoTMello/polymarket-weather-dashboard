#!/bin/bash
cd /home/frederico/.openclaw/workspace/skills/polymarket-weather
while true; do
    echo "[$(date)] Starting server..."
    python3 server.py
    echo "[$(date)] Server died, restarting in 3s..."
    sleep 3
done
