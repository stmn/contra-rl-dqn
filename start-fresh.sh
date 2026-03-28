#!/bin/bash
PORT=41918

# Kill previous instance
lsof -ti:$PORT | xargs kill -9 2>/dev/null && echo "Previous instance killed" || echo "No previous instance"

# Clean everything — fresh start
rm -rf checkpoints/ logs/stats.json

# Activate venv and start
source .venv/bin/activate
DEVICE=mps python scripts/run.py
