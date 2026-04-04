# Plan: Extra Lives Exploration

## Idea

Give the agent 30 lives (`$32 = 29`) instead of 3 so it can explore further sections of the map and collect experiences. Then restore 3 lives for actual training.

## RAM

- `$32` = Player 1 lives (0 = last life, 2 = 3 lives standard)
- `$38` = P1 Game Over Status (0 = playing, 1 = game over)
- Source: https://datacrystal.tcrf.net/wiki/Contra_(NES)/RAM_map

## Variants

### A. Permanent 30 lives
- `nes[0x32] = 29` after every reset
- Agent always has 30 lives, reaches further
- Problem: death penalty -500 × 30 = -15000, total reward negative
- Solution: reduce or disable death penalty

### B. Exploration + training
- Episode with 30 lives WITHOUT training (exploration only, collect experiences to buffer)
- At random moment: restore 3 lives (`nes[0x32] = 2`), ENABLE training
- Buffer has experiences from the entire map

### C. Konami Code
- Alternative: input Up,Up,Down,Down,Left,Right,Left,Right,B,A on title screen
- Gives 30 lives without RAM modification
- Requires going through title screen (slower boot)

## Implementation (RAM)
```python
# In ContraEnv._boot() or reset(), after loading initial_state:
if settings.extra_lives:
    self._nes[0x32] = 29  # 30 lives
```

## Status
Future plan. To implement when agent stagnates and needs experiences from later map sections.
