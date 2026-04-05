# Plan: Buffer Persistence & Cold Start Protection

## Problem

After every restart, replay buffer is empty. Model weights are preserved from checkpoint but training on new low-quality data from empty buffer causes performance collapse ("cold buffer problem" / distribution mismatch).

Noisy nets amplify the issue — noise causes random exploration, buffer fills with bad transitions, agent trains on them and spirals down.

## Root Cause

Network weights were calibrated to transitions from experienced play. Empty buffer fills with exploratory/noisy data that looks completely different. Training on this unrepresentative data shifts weights away from the good solution. This is the same distribution shift that experience replay was designed to prevent.

## Solutions (ordered by effectiveness)

### 1. Save/Restore Replay Buffer (Best)
Save buffer contents to disk alongside model checkpoint. Reload on restart.

- Standard buffer: pickle or numpy arrays
- PER: also save SumTree priorities
- Size: 100K entries × ~128KB = ~12GB. Compression helps (NES frames compress well)
- Even partial save (50K entries) helps enormously

### 2. Freeze Training During Warm-Up (Simplest)
After loading checkpoint, fill buffer without gradient updates:

```python
# In DQN trainer, after loading checkpoint:
if len(self.replay_buffer) < self.learning_starts:
    # Play with loaded policy (greedy/low noise) but don't train
    pass  # skip _train_step() until buffer has enough data
```

Already partially implemented via `learning_starts = 1000`, but 1000 may be too low. Consider 10K-50K.

### 3. Reset NoisyNet Sigma on Resume
After loading checkpoint, set noise parameters to near-zero so agent plays near-greedy while buffer fills:

```python
for m in self.q_network.modules():
    if isinstance(m, NoisyLinear):
        m.weight_sigma.data.fill_(0.01)  # very low noise
        m.bias_sigma.data.fill_(0.01)
```

Gradually increase sigma as buffer fills.

### 4. Lower Learning Rate on Resume
Use reduced LR for first N updates. Less destructive gradient updates during distribution mismatch.

### 5. Reduce Update Frequency on Resume
Train every 16 steps instead of 4 until buffer is sufficiently full. More data per gradient step = more stable.

## References

- Fedus et al. (2020), "Revisiting Fundamentals of Experience Replay" — buffer composition critically affects stability
- Nikishin et al. (2022), "The Primacy Bias in Deep RL" — early training data has outsized influence
- Mnih et al. (2015), DQN paper — warm-up phase (50K random frames) exists for this exact reason

## Status
Not implemented. Apply when ready to minimize impact of restarts.
