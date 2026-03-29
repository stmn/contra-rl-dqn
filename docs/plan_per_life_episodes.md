# Plan: Per-Life Episodes

## Idea
Each life = separate episode. Death ends the episode but game continues from respawn point instead of restarting from level start.

## Current behavior
- 1 episode = 3 lives (full game until game over)
- Death gives -100 penalty but episode continues
- Agent accumulates reward across all 3 lives
- Death penalty is ~1.6% of total reward — barely noticeable

## Proposed behavior
- 1 episode = 1 life
- Death = episode end (no more reward)
- Next episode starts from respawn point (not level start)
- Agent sees immediate consequence of dying

## Example
- Life 1: 3000 scroll → dies → episode reward: 3000
- Life 2: from respawn, 2000 scroll → dies → episode reward: 2000
- Life 3: from respawn, 1500 scroll → dies → episode reward: 1500
- Agent clearly sees life 1 was better than life 3

## Why it should help
- Death = end of episode = zero future reward (much stronger signal than -100)
- Q-values learn per-situation, not averaged across 3 lives
- Agent sees more diverse starting positions (respawn points)
- Faster feedback loop (shorter episodes)

## Risks
- reset() without restarting the game — need to only zero reward counters, not reload savestate
- Cumulative scroll tracker needs to handle mid-game reset correctly
- After 3rd death (game over), must reload savestate for real restart
- Replay buffer will have experiences from different starting positions — could help or hurt generalization

## Implementation notes
- On death ($90 == 2): return terminated=True but don't reload savestate
- On game over ($38 == 1): return terminated=True AND reload savestate
- Reset only zeroes: _step_count, _total_reward, _prev_scroll, _idle_counter
- Keep _cumulative_scroll running across lives (don't reset)
- Track which life we're on (1, 2, 3) for debugging
