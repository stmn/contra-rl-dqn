# Plan: Prioritised Experience Replay (PER)

## Current behavior
- Replay buffer (100K) samples uniformly at random
- Every experience has equal chance of being used for training
- Rare successful dodges have 0.001% chance of being sampled — same as routine "run right" steps

## Proposed behavior
- Each experience gets a **priority** based on TD error (how much it "surprised" the network)
- Higher surprise = sampled more often for training
- Agent accidentally dodges a bullet → network predicted death but agent survived → high TD error → high priority → sampled 10-100x more often

## How it works
1. On each training step, compute TD error: `|predicted Q - actual reward + γ * next Q|`
2. Store this error as priority for the experience
3. When sampling batch, use priority-weighted probability instead of uniform random
4. Apply importance sampling weights to correct for the bias introduced by non-uniform sampling

## Why it should help
- Agent in Contra rarely dodges successfully — when it does, that experience should be learned from repeatedly
- Currently those rare moments drown in 100K buffer of "run right and shoot" steps
- PER was used in the academic paper "General Deep RL in NES Games" which achieved the best NES results

## Implementation
- Change `ReplayBuffer` to store priorities per experience
- Use `SumTree` data structure for efficient priority-weighted sampling (O(log n))
- Add `alpha` parameter (0.6 typical): how much prioritization to use (0 = uniform, 1 = full priority)
- Add `beta` parameter (0.4 → 1.0): importance sampling correction, annealed during training
- Update priorities after each training step with new TD errors

## Reference
- Original paper: "Prioritized Experience Replay" (Schaul et al., 2015) https://arxiv.org/abs/1511.05952
- Used in "General Deep RL in NES Games" (LeBlanc & Lee, 2021) — beat Super Mario Bros level 1

## Risk
- More complex code
- Slightly slower training (priority updates + tree sampling)
- Hyperparameters alpha/beta need tuning
