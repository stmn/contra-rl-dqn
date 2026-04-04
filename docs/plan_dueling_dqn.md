# Plan: Dueling DQN

## Idea

Split Q-value into two streams:
- **V(s)** — state value (how good is this situation regardless of action)
- **A(s,a)** — action advantage (how good is this action vs average)

Q(s,a) = V(s) + A(s,a) - mean(A)

The agent learns that some states are bad regardless of action (e.g., standing in front of a bullet = low V). It doesn't need to try all 16 actions to discover this.

## Code change

Only the network head — CNN and features_net unchanged:

```python
# Before:
self.head = nn.Sequential(
    nn.Linear(512 + 32, 256),
    nn.ReLU(),
    nn.Linear(256, n_actions),
)

# Dueling:
self.value_stream = nn.Sequential(
    nn.Linear(512 + 32, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
)
self.advantage_stream = nn.Sequential(
    nn.Linear(512 + 32, 256),
    nn.ReLU(),
    nn.Linear(256, n_actions),
)

def forward(self, ...):
    combined = ...  # CNN + features (unchanged)
    value = self.value_stream(combined)
    advantage = self.advantage_stream(combined)
    q = value + advantage - advantage.mean(dim=1, keepdim=True)
    return q
```

## Impact
- ~20 lines of code change in HybridQNetwork/QNetwork
- Requires retraining (different architecture)
- Confirmed 20-30% better results on Atari (Wang et al., 2016)
- Zero impact on rest of pipeline

## Rainbow DQN progress (6 extensions)
- [x] Double DQN
- [x] Prioritised Experience Replay (PER)
- [x] **Dueling DQN** ← implemented
- [x] N-step returns
- [x] Noisy nets
- [ ] Distributional RL (C51)

## Status
Implemented. Feature flag: `DUELING_DQN=true`

## Source
Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016
https://arxiv.org/abs/1511.06581
