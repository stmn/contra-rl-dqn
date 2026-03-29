# Plan: Empirical Floor Map ("Paint Footprints")

## Idea
Instead of decoding complex tile collision data from RAM, build a floor map empirically by tracking where the player walks. Like boots dipped in paint on invisible floor — every step marks that position as "walkable ground."

## How it works
1. Create 2D grid in world coordinates (level_width × screen_height)
2. Every frame where player is on ground (not jumping): mark (world_x, player_y) as floor
3. world_x = cumulative_scroll + player_screen_x ($334)
4. player_y = $31A
5. Ground detection: jump status ($A0) == 0 and Y velocity ($C6) == 0
6. Map persists across episodes and grows over time
7. Save map to disk periodically so it survives restarts

## After 1000+ episodes
- Map shows exactly where platforms, holes, water are
- No RAM tile decoding needed — pure empirical data
- More accurate than RAM interpretation (captures actual walkable areas)

## Usage for agent

### Option A: Visual overlay
- Render nearby map section as thin green line on game frame
- Agent sees floor ahead through CNN

### Option B: Feature vector (for hybrid observation)
- Sample map at positions ahead of player: (x+16, y), (x+32, y), (x+48, y)...
- If mapped = 1, if unknown/hole = 0
- "Is there floor 32px ahead? Yes/No"
- Feed as part of RAM features vector

### Option C: Separate input channel
- Render map slice as small 1D array (floor/no floor for next 128px ahead)
- Feed alongside image and RAM features

## Data structure
- NumPy array: shape (max_world_x, 240), dtype uint8
- Value: 0 = unknown, 1 = confirmed floor
- Level 1 of Contra is ~3000px wide → array of ~3000 × 240 = ~700KB
- Tiny memory footprint

## Persistence
- Save to `logs/floor_map.npy` every 100 episodes
- Load on startup if exists
- Reset on `start-fresh.sh`

## Advantages
- Zero RAM decoding complexity
- Gets more accurate with time
- Shows ACTUAL walkable areas (not theoretical collision tiles)
- Simple to implement (few lines of code)
- Can visualize as debug overlay to verify correctness

## Risks
- Early training: map is empty, no useful data yet
- Player might never walk on some platforms → holes in map
- Jump arcs don't get mapped (only ground contact points)
- Map is level-specific — resets if game changes level
