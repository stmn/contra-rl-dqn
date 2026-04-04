# Plan: BTR (Beyond The Rainbow) Upgrades

Based on "Beyond The Rainbow: High Performance Deep RL on a Desktop PC" (Clark et al., 2024)
Paper: https://arxiv.org/abs/2411.03820
Reference implementation: https://github.com/VIPTankz/BTR

## Overview

5 upgrades from BTR paper, ordered by implementation priority. Each behind a feature flag.
To implement after testing current Rainbow DQN setup on Mac Mini.

## 1. Hyperparameters (zero code, config only)

Changes to `config/settings.py` defaults and DQN trainer:

| Parameter | Current | BTR | Rationale |
|-----------|---------|-----|-----------|
| gamma | 0.99 | 0.997 | Prevents optimal agents from having suboptimal policies due to excessive discounting |
| target_update_freq | 1,000 | 500 | Better stability with larger updates |
| PER alpha | 0.6 | 0.2 | Lower prioritization recommended when using IQN |
| Exploration | Noisy only | Noisy + epsilon until 100M, then eps=0 | Dual exploration |

**Complexity:** Trivial — config changes only.

## 2. Spectral Normalization

Apply `torch.nn.utils.spectral_norm()` to all convolutional layers. Constrains Lipschitz constant, stabilizes training.

```python
# Before:
nn.Conv2d(4, 32, 8, stride=4)

# After:
spectral_norm(nn.Conv2d(4, 32, 8, stride=4))
```

Flag: `spectral_norm: bool = True`

**Complexity:** Low — one wrapper per conv layer (~5 lines).

## 3. IMPALA CNN Architecture

Replace 3-layer CNN with IMPALA residual network (Espeholt et al., 2018) scaled 2x width + adaptive max pooling.

```
Current: Conv(4→32,8x8,s4) → Conv(32→64,4x4,s2) → Conv(64→64,3x3,s1) → Linear(9216→512)

IMPALA:  3 residual blocks, each:
         Conv(channels) → MaxPool(3x3,s2) → ResBlock → ResBlock
         Block 1: 32 channels (×2 = 64)
         Block 2: 64 channels (×2 = 128)
         Block 3: 64 channels (×2 = 128)
         → AdaptiveMaxPool2d(6,6) → Linear(128*6*6→512)
```

Flag: `impala_cnn: bool = True`

**Impact:** +142% IQM in ablation — single largest contributor.
**Complexity:** Moderate — ~60 lines new architecture class. Well-documented in PyTorch.

## 4. Munchausen RL

Add scaled log-policy to bootstrap target. Converts DQN to Soft-DQN.

Target becomes:
```
r_t + alpha * tau * ln(pi(a_t|s_t)) + gamma * sum_a' pi(a'|s_{t+1}) * (Q(s_{t+1}, a') - tau * ln(pi(a'|s_{t+1})))
```

Where pi = softmax(Q/tau), alpha=0.9, tau=0.03.

**Replaces Double DQN** (no argmax over next state).
Reduces policy churn from 11% to 3.8%.

Flag: `munchausen_rl: bool = True`

**Complexity:** Moderate — ~30 lines in loss/target calculation. Must clip log-policy to avoid -inf.

## 5. IQN (Implicit Quantile Networks)

Replace single Q-value per action with full quantile distribution.

- Sample random quantiles tau ~ U(0,1), typically 8 samples
- Cosine embedding: phi(tau) = ReLU(Linear(cos(pi * i * tau)))
- Quantile Q-values: Q(s,a,tau) = network(s) * phi(tau)
- Loss: quantile Huber loss across tau samples

**Replaces C51** (which we don't have — this would be new distributional RL).

Flag: `iqn: bool = True`

**Complexity:** High — ~100 lines. New network forward pass, new loss function, cosine embedding layer. Significant refactor of `_train_step`.

## Implementation order

1. Hyperparameters — immediate, no risk
2. Spectral Normalization — trivial, test stability
3. IMPALA CNN — biggest impact, moderate effort
4. Munchausen RL — replaces Double DQN, strong stability gain
5. IQN — biggest code change, implement last

## Files to modify

- `config/settings.py` — new feature flags
- `contra/training/dqn.py` — all changes (networks, loss, target)

## Status

Planned. Implement after current Rainbow DQN has been evaluated on Mac Mini (~1 week training).
