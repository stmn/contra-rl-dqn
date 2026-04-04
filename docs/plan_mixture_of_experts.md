# Plan: Mixture of Experts — Segments

## Idea

Split a level into segments. Each segment has its own model (network + replay buffer). Router switches based on scroll position. Agent trains all segments simultaneously.

## Key problem: rare segments

If the agent reaches the boss in 5% of episodes, the boss model gets 20x fewer experiences. Solution: **auto-collecting Save States at segment boundaries**.

### Auto Save States
- When agent crosses a segment boundary → NES state (`nes.save()`, ~28KB) saved to pool
- On respawn in segment → random state from pool (varied starting situations)
- 100 states = ~2.7MB, 1000 = ~27MB — no limit needed
- Agent sees different paths/weapons/positions instead of one frozen moment

## Architecture

### Segments (markers on map)
```
segments = [
    {"scroll": 30000},  # boundary: start → 30K = Segment 0
    {"scroll": 60000},  # boundary: 30K → 60K = Segment 1
    {"scroll": 90000},  # boundary: 60K → boss = Segment 2
]
# Segment 0: start to 30K
# Segment 3: 90K to end of level (boss)
```
- Set by user in dashboard (click on progress bar or enter scroll value)
- Visible on Level Progress bar as vertical lines
- No segments = single model (current behavior)

### SegmentManager (`contra/training/segments.py`)
- N instances of QNetwork/HybridQNetwork + N target networks + N replay buffers
- Save State pool per segment (auto-collected at boundaries)
- `get_segment(scroll)` → segment index
- `select_action(obs, segment_idx, epsilon)` → forward through correct model
- `store(segment_idx, transition)` → to correct buffer
- `train_step(segment_idx)` → train correct model
- `save(dir)` / `load(dir)` → all models + states

### DQN Trainer changes
- Delegate to SegmentManager instead of direct q_network/buffer
- Epsilon, global_step — shared
- Auto-rollback per segment

### Dashboard
- **Segments tab** (replaces Practice):
  - Segment list with scroll position, buffer size, status
  - Add/Remove segment
  - Save State pool per segment (count collected)
- **Level Progress bar** — segment markers as vertical lines
- Active segment highlighted

### Respawn flow
1. Agent dies → episode ends
2. New episode: choose segment to train (round-robin or priority)
3. Load random Save State from chosen segment's pool
4. Or: play from start (segment 0) — models switch automatically

### "All at once" mode
Agent plays from start. Each segment uses its own model. Experiences go to the correct buffer. At boundaries, auto-save state to pool. If agent dies in segment 2, segment 2's model learns. Segment 0's model also learned (from the start of the same run).

## Challenges

1. **Knowledge transfer** — "dodge bullets" must be learned in each model
   - Solution: pre-train on segment 0, copy weights as starting point for new segments
2. **Boundaries** — smooth handoff between models
   - Overlap zone (200px) with Q-value interpolation
3. **RAM** — N models × ~77MB = a lot
   - Models are small (~2-5MB weights), 77MB is optimizer state
   - 5 segments × 5MB = 25MB weights, acceptable
4. **Save State timing** — state saved at boundary may not be ideal
   - Pool of 100+ states mitigates this — diversity

## Status

Partially implemented — per-level models exist. Full segment system (within a level) not yet built. Requires:
- `contra/training/segments.py` — SegmentManager
- Refactor DQN trainer
- Dashboard Segments tab
- API endpoints
