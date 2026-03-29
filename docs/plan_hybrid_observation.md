# Plan: Hybrid Observation (Pixels + RAM Features)

## Problem
CNN must detect 2-3 pixel bullets on 128x128 image, determine direction, assess threat, choose dodge action — all from raw pixels. This is extremely hard and may never work reliably.

## Insight
We already read enemy positions, velocities, and player state from RAM (for overlay). But this data goes onto the IMAGE and CNN must re-extract it from pixels. Pointless bottleneck — feed the data directly to the network.

## Proposed architecture

```
image (128x128x4) → CNN → 512 features ─┐
                                          ├→ concat (527) → Dense 512 → 16 Q-values
RAM vector (15 numbers) → Dense 32 ──────┘
```

CNN handles NAVIGATION (platforms, terrain, where to go).
RAM vector handles SURVIVAL (where are threats, how close, dodge?).

## RAM features vector (15 values, all normalized 0-1)

From player:
1. Player X on screen (0x334) / 256
2. Player Y on screen (0x31A) / 240
3. Player state: 1 if alive, 0 if dead/respawn (0x90)
4. Player in air: from jump status (0xA0)

From enemies (nearest 3):
5-6. Enemy 1: distance X, distance Y (relative to player, normalized)
7. Enemy 1: X velocity (0x508) / 256
8. Enemy 1: attack delay timer (0x558) / 256 — how soon until next shot
9-10. Enemy 2: distance X, distance Y
11. Enemy 2: X velocity
12. Enemy 2: attack delay timer
13-14. Enemy 3: distance X, distance Y
15. Enemy 3: X velocity
16. Enemy 3: attack delay timer

Aggregate:
17. Number of enemies on screen / 16
18. Nearest enemy distance (euclidean) / 256
19. Enemy attack flag (0x8E) — are enemies actively shooting

From player movement:
20. Edge fall code (0xA4) — about to fall off left/right edge
21. Jump status (0xA0) — in air or on ground
22. Y velocity (0xC6) — going up or down

## What changes in code

### 1. ContraEnv
- `step()` returns observation as dict: `{"image": np.array(128,128,4), "features": np.array(15)}`
- New method `_build_features()` reads RAM, calculates distances, normalizes
- Observation space changes to `gymnasium.spaces.Dict`

### 2. QNetwork (dqn.py)
- Two input paths: CNN for image, small Dense for features
- Concatenate before final Dense layers
- Forward takes (image_tensor, features_tensor)

### 3. ReplayBuffer
- Stores tuples: (image, features, action, reward, next_image, next_features, done)
- Slightly more memory but features are tiny (15 floats vs 128x128 image)

### 4. DQNTrainer
- `_select_action()` passes both image + features to network
- `_train_step()` samples both from buffer, passes both to network
- Epsilon-greedy unchanged

### 5. Wrappers
- StreamCapture passes features alongside frame
- FrameStack only stacks images, features are current-only (no stacking needed)

### 6. Dashboard
- Can display features vector as debug overlay (optional)

## Why this solves multiple problems

1. **Bullet detection** — positions from RAM, no pixel detection needed
2. **Reaction time** — numeric data processed instantly, no CNN bottleneck
3. **Overlay artifacts** — overlay becomes optional/cosmetic, not functional
4. **Resolution** — less important since critical data bypasses image
5. **Frame skip** — even with skip 2, agent gets exact enemy positions

## Why it's not cheating
- Data from RAM is information ALREADY VISIBLE on screen to human player
- Human instantly knows "enemy is close on the right" — agent gets same awareness in numeric form
- Agent still must LEARN what to do with the information

## Risk
- Dict observation space is less standard — may need wrapper adjustments
- Feature engineering (choosing which 15 values) could miss important info
- Agent might ignore image entirely and rely only on features (fine for combat, bad for navigation)

## Mitigation for "ignoring image"
- Features only contain enemy/threat data, NOT terrain/platform data
- Agent MUST use image for navigation — features don't tell where platforms are
- Natural separation: image = where to go, features = what to avoid
