"""Generate NES save states for start of each level.

Uses Konami Code (30 lives) and runs right + shoots through each level.
Saves states when RAM $30 (level number) changes.

Usage: python scripts/gen_level_states.py
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cynes

STATES_DIR = Path("checkpoints/level_states")
STATES_DIR.mkdir(parents=True, exist_ok=True)

nes = cynes.NES(str(Path("roms/contra.nes")))
nes.reset()

# Wait for title screen
for _ in range(300):
    nes.step(1)

# Konami Code: Up Up Down Down Left Right Left Right B A
konami = [
    (cynes.NES_INPUT_UP, 3), (0, 3),
    (cynes.NES_INPUT_UP, 3), (0, 3),
    (cynes.NES_INPUT_DOWN, 3), (0, 3),
    (cynes.NES_INPUT_DOWN, 3), (0, 3),
    (cynes.NES_INPUT_LEFT, 3), (0, 3),
    (cynes.NES_INPUT_RIGHT, 3), (0, 3),
    (cynes.NES_INPUT_LEFT, 3), (0, 3),
    (cynes.NES_INPUT_RIGHT, 3), (0, 3),
    (cynes.NES_INPUT_B, 3), (0, 3),
    (cynes.NES_INPUT_A, 3), (0, 3),
]
for btn, frames in konami:
    nes.controller = btn
    for _ in range(frames):
        nes.step(1)

# Press Start
nes.controller = cynes.NES_INPUT_START
for _ in range(5):
    nes.step(1)
nes.controller = 0
for _ in range(60):
    nes.step(1)

# Select 1 Player
nes.controller = cynes.NES_INPUT_UP
for _ in range(5):
    nes.step(1)
nes.controller = 0
for _ in range(15):
    nes.step(1)
nes.controller = cynes.NES_INPUT_START
for _ in range(5):
    nes.step(1)
nes.controller = 0
for _ in range(600):
    nes.step(1)

# Verify 30 lives
lives = nes[0x32]
print(f"Lives: {lives + 1} ({'30 lives OK' if lives >= 28 else 'Konami Code FAILED'})")

# Save Level 1 state
state = nes.save()
np.save(str(STATES_DIR / "level_0.npy"), state)
print(f"Level 1 saved ({len(state)} bytes)")

# Play through levels — run right + jump + shoot
current_level = 0
saved_levels = {0}
frame = 0
MAX_FRAMES = 500_000  # ~2.5 hours at 60fps, should be enough for 8 levels

RIGHT_JUMP_SHOOT = cynes.NES_INPUT_RIGHT | cynes.NES_INPUT_A | cynes.NES_INPUT_B

while frame < MAX_FRAMES:
    nes.controller = RIGHT_JUMP_SHOOT
    nes.step(1)
    frame += 1

    level = nes[0x30]

    if level != current_level and level not in saved_levels and level <= 7:
        # New level! Wait a moment for it to fully load
        for _ in range(300):
            nes.step(1)
            frame += 1

        state = nes.save()
        np.save(str(STATES_DIR / f"level_{level}.npy"), state)
        saved_levels.add(level)
        print(f"Level {level + 1} saved at frame {frame}")
        current_level = level

        if len(saved_levels) == 8:
            print("All 8 levels saved!")
            break

    if frame % 50000 == 0:
        print(f"Frame {frame}, level {current_level + 1}, lives {nes[0x32] + 1}")

print(f"\nDone. Saved {len(saved_levels)} level states in {STATES_DIR}")
for f in sorted(STATES_DIR.glob("level_*.npy")):
    print(f"  {f.name}")
