# Plan: Dueling DQN

## Pomysł

Rozdzielić Q-value na dwa strumienie:
- **V(s)** — wartość stanu (jak dobra jest ta sytuacja niezależnie od akcji)
- **A(s,a)** — przewaga akcji (jak dobra jest ta akcja vs średnia)

Q(s,a) = V(s) + A(s,a) - mean(A)

Agent uczy się że niektóre stany są złe niezależnie od akcji (np. stanie przed pociskiem = niskie V). Nie musi próbować wszystkich 16 akcji żeby to odkryć.

## Zmiana w kodzie

Tylko head sieci — CNN i features_net bez zmian:

```python
# Teraz:
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
    combined = ...  # CNN + features (bez zmian)
    value = self.value_stream(combined)
    advantage = self.advantage_stream(combined)
    q = value + advantage - advantage.mean(dim=1, keepdim=True)
    return q
```

## Wpływ
- ~20 linii kodu zmiany w HybridQNetwork/QNetwork
- Wymaga retreningu (inna architektura)
- Potwierdzone 20-30% lepsze wyniki na Atari (Wang et al., 2016)
- Zero wpływu na resztę pipeline

## Co mamy z Rainbow DQN (6 ulepszeń)
- [x] Double DQN
- [x] Prioritised Experience Replay (PER)
- [ ] **Dueling DQN** ← ten plan
- [ ] N-step returns
- [ ] Noisy nets (exploration bez epsilon)
- [ ] Distributional RL (C51)

## Status
Do wdrożenia. Najniższy koszt, najwyższy oczekiwany zysk.

## Źródło
Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016
https://arxiv.org/abs/1511.06581
