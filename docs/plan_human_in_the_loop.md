# Plan: Human-in-the-Loop Action Correction

## Idea
Pause training at critical moment, choose the correct action manually, agent executes it and receives extra bonus reward. The experience enters replay buffer with high reward, teaching the agent "in THIS situation, do THIS."

## How it works
1. User watches dashboard, sees agent about to make bad decision
2. Clicks Pause
3. Selects action from UI (e.g., "Jump", "Duck+Shoot", "Jump Right")
4. Clicks Unpause — agent executes chosen action for N frames
5. Experience stored in replay buffer with large bonus reward
6. With PER (prioritised replay), this experience gets sampled hundreds of times

## Why 5 corrections can matter
- Without PER: 5 entries in 100K buffer = 0.005% chance of sampling = no impact
- With PER: 5 entries with highest TD error = sampled 100x more often = effectively 500 entries = real impact
- **Requires PER to be effective**

## Dashboard UI changes
- When paused, show action selector (16 buttons or simplified groups)
- Groups: "Jump", "Jump Right", "Duck", "Duck+Shoot", "Shoot Up", etc.
- Selected action highlighted, executes on unpause
- Visual indicator: "HUMAN ACTION" overlay during forced action

## Implementation
1. Add `forced_action` field to `TrainingControls`
2. When paused + action selected + unpaused: DQN uses forced_action instead of epsilon-greedy for N steps
3. Bonus reward multiplier (e.g., 5x normal reward) for forced action steps
4. Experience enters replay buffer normally (but with inflated reward → high TD error → high PER priority)

## Dependency
- **Requires PER (plan_prioritised_replay.md)** to be effective
- Without PER, human corrections drown in 100K buffer noise

## Risks
- Wrong human action = teaching bad behavior with high priority
- Bonus reward too high = distorts Q-value scale
- Pausing at wrong moment = agent in mid-action, forced action doesn't make sense
- Time investment: even 5 corrections require watching and timing carefully
