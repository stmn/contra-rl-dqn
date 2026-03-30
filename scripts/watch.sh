#!/bin/bash
# Watch the trained agent play Contra with sound
cd "$(dirname "$0")/.."
echo "Starting FCEUX..."
fceux --loadlua scripts/fceux_agent.lua --palette roms/cynes.pal roms/contra.nes &
FCEUX_PID=$!
echo "Select 1 Player in FCEUX, then agent takes over."
echo ""
.venv/bin/python scripts/watch.py "$@"
kill $FCEUX_PID 2>/dev/null
