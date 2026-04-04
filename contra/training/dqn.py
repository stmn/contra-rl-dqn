"""DQN with Rainbow extensions: Double DQN, PER, Dueling, Noisy nets, N-step, Huber loss.

Feature flags (from settings/.env):
- HYBRID_OBSERVATION, PRIORITISED_REPLAY, DUELING_DQN, NOISY_NETS
- N_STEP_RETURNS, HUBER_LOSS, GRADIENT_CLIP
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import settings


# ============================================================
# Noisy Linear Layer (Fortunato et al., 2018)
# ============================================================

class NoisyLinear(nn.Module):
    """Linear layer with learnable noise for exploration."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


def _make_linear(in_f: int, out_f: int, noisy: bool) -> nn.Module:
    return NoisyLinear(in_f, out_f) if noisy else nn.Linear(in_f, out_f)


# ============================================================
# Networks
# ============================================================

class QNetwork(nn.Module):
    """Standard CNN Q-network (image only)."""

    def __init__(self, n_actions: int, dueling: bool = False, noisy: bool = False) -> None:
        super().__init__()
        self._dueling = dueling
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            _make_linear(64 * 12 * 12, 512, noisy), nn.ReLU(),
        )
        if dueling:
            self.value_stream = nn.Sequential(_make_linear(512, 256, noisy), nn.ReLU(), _make_linear(256, 1, noisy))
            self.advantage_stream = nn.Sequential(_make_linear(512, 256, noisy), nn.ReLU(), _make_linear(256, n_actions, noisy))
        else:
            self.head = nn.Sequential(_make_linear(512, 256, noisy), nn.ReLU(), _make_linear(256, n_actions, noisy))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x / 255.0)
        if self._dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.head(features)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class HybridQNetwork(nn.Module):
    """CNN + RAM features Q-network."""

    def __init__(self, n_actions: int, n_features: int = 28, dueling: bool = False, noisy: bool = False) -> None:
        super().__init__()
        self._dueling = dueling
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            _make_linear(64 * 12 * 12, 512, noisy), nn.ReLU(),
        )
        self.features_net = nn.Sequential(
            _make_linear(n_features, 64, noisy), nn.ReLU(),
            _make_linear(64, 32, noisy), nn.ReLU(),
        )
        combined_size = 512 + 32
        if dueling:
            self.value_stream = nn.Sequential(_make_linear(combined_size, 256, noisy), nn.ReLU(), _make_linear(256, 1, noisy))
            self.advantage_stream = nn.Sequential(_make_linear(combined_size, 256, noisy), nn.ReLU(), _make_linear(256, n_actions, noisy))
        else:
            self.head = nn.Sequential(_make_linear(combined_size, 256, noisy), nn.ReLU(), _make_linear(256, n_actions, noisy))

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(image / 255.0)
        feat_out = self.features_net(features)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        if self._dueling:
            value = self.value_stream(combined)
            advantage = self.advantage_stream(combined)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.head(combined)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ============================================================
# Replay Buffers
# ============================================================

class ReplayBuffer:
    """Uniform random replay buffer."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        return self._unpack(batch), None, None  # no weights, no indices

    def update_priorities(self, indices, priorities):
        pass  # no-op for uniform buffer

    def _unpack(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, np.array(actions), np.array(rewards, dtype=np.float32), next_states, np.array(dones, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buf)


class NStepBuffer:
    """Accumulates N transitions and computes N-step return."""

    def __init__(self, n: int = 3, gamma: float = 0.99) -> None:
        self._n = n
        self._gamma = gamma
        self._buf: deque = deque(maxlen=n)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def get(self):
        """Returns N-step transition or None if not ready."""
        if len(self._buf) < self._n:
            return None
        # Compute N-step return
        state = self._buf[0][0]
        action = self._buf[0][1]
        n_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self._buf):
            n_reward += (self._gamma ** i) * r
            if d:
                return (state, action, n_reward, self._buf[i][3], True)
        return (state, action, n_reward, self._buf[-1][3], self._buf[-1][4])

    def flush(self):
        """Flush remaining transitions at episode end."""
        results = []
        while len(self._buf) > 0:
            state = self._buf[0][0]
            action = self._buf[0][1]
            n_reward = 0.0
            for i, (_, _, r, _, d) in enumerate(self._buf):
                n_reward += (self._gamma ** i) * r
                if d:
                    results.append((state, action, n_reward, self._buf[i][3], True))
                    break
            else:
                results.append((state, action, n_reward, self._buf[-1][3], self._buf[-1][4]))
            self._buf.popleft()
        return results

    def reset(self):
        self._buf.clear()


class SumTree:
    """Binary tree for O(log n) priority-weighted sampling."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data = [None] * capacity
        self._write = 0
        self._size = 0

    def add(self, priority: float, data) -> None:
        idx = self._write + self._capacity - 1
        self._data[self._write] = data
        self._update(idx, priority)
        self._write = (self._write + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def _update(self, idx: int, priority: float) -> None:
        change = priority - self._tree[idx]
        self._tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self._tree[idx] += change

    def get(self, s: float) -> tuple[int, float, object]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self._tree):
                break
            if s <= self._tree[left] or right >= len(self._tree):
                idx = left
            else:
                s -= self._tree[left]
                idx = right
        data_idx = idx - self._capacity + 1
        return idx, self._tree[idx], self._data[data_idx]

    @property
    def total(self) -> float:
        return self._tree[0]

    @property
    def max_priority(self) -> float:
        return max(self._tree[self._capacity - 1:self._capacity - 1 + self._size]) if self._size > 0 else 1.0

    def __len__(self) -> int:
        return self._size


class PrioritisedReplayBuffer:
    """Prioritised Experience Replay using SumTree."""

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta_start: float = 0.4,
                 total_steps: int = 100_000_000, train_freq: int = 4) -> None:
        self._tree = SumTree(capacity)
        self._alpha = alpha
        self._beta = beta_start
        n_training_steps = total_steps // train_freq
        self._beta_increment = (1.0 - beta_start) / max(n_training_steps, 1)
        self._epsilon = 1e-6

    def push(self, state, action, reward, next_state, done) -> None:
        # max_priority is already alpha-exponentiated from update_priorities(), use directly
        priority = self._tree.max_priority if len(self._tree) > 0 else 1.0
        self._tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = []
        indices = []
        priorities = []
        segment = self._tree.total / batch_size

        self._beta = min(1.0, self._beta + self._beta_increment)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self._tree.get(s)
            if data is None:
                # Fallback: resample
                s = random.uniform(0, self._tree.total)
                idx, priority, data = self._tree.get(s)
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)

        if len(batch) < batch_size:
            return None, None, None

        states, actions, rewards, next_states, dones = zip(*batch)
        priorities_arr = np.array(priorities, dtype=np.float64) + self._epsilon
        probs = priorities_arr / self._tree.total
        weights = (len(self._tree) * probs) ** (-self._beta)
        weights = weights / weights.max()

        return (
            (states, np.array(actions), np.array(rewards, dtype=np.float32), next_states, np.array(dones, dtype=np.float32)),
            np.array(weights, dtype=np.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors) -> None:
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self._epsilon) ** self._alpha
            self._tree._update(idx, priority)

    def __len__(self) -> int:
        return len(self._tree)


# ============================================================
# Trainer
# ============================================================

class DQNTrainer:
    """DQN training loop with optional hybrid obs and PER."""

    def __init__(
        self,
        env,
        *,
        device: str | None = None,
        on_step: Callable[[int, float], None] | None = None,
        on_episode: Callable[[float, dict], None] | None = None,
        controls=None,
        on_save: Callable[[], None] | None = None,
        frame_buffer=None,
    ) -> None:
        self.env = env
        self.device = torch.device(device or settings.device)
        self.on_step = on_step
        self.on_episode = on_episode
        self.controls = controls
        self.on_save = on_save
        self.frame_buffer = frame_buffer

        self.n_actions = env.action_space.n
        self.total_timesteps = settings.total_timesteps
        self._hybrid = settings.hybrid_observation
        self._per = settings.prioritised_replay
        self._dueling = settings.dueling_dqn
        self._noisy = settings.noisy_nets
        self._n_step = settings.n_step_returns
        self._huber = settings.huber_loss
        self._grad_clip = settings.gradient_clip

        # Hyperparameters
        self.lr = 1e-4
        self.gamma = 0.99
        self.batch_size = 32
        self.buffer_size = 100_000
        self.learning_starts = 1_000
        self.train_freq = 4
        self.target_update_freq = 1_000
        self.epsilon_start = 1.0 if not self._noisy else 0.0  # noisy nets handle exploration
        self.epsilon_end = 0.05 if not self._noisy else 0.0
        self.epsilon_decay = 50_000

        # Networks
        net_kwargs = {"dueling": self._dueling, "noisy": self._noisy}
        if self._hybrid:
            self.q_network = HybridQNetwork(self.n_actions, **net_kwargs).to(self.device)
            self.target_network = HybridQNetwork(self.n_actions, **net_kwargs).to(self.device)
        else:
            self.q_network = QNetwork(self.n_actions, **net_kwargs).to(self.device)
            self.target_network = QNetwork(self.n_actions, **net_kwargs).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Replay buffer
        if self._per:
            self.replay_buffer = PrioritisedReplayBuffer(
                self.buffer_size, total_steps=self.total_timesteps, train_freq=self.train_freq)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)

        # N-step buffer
        self._nstep_buf = NStepBuffer(self._n_step, self.gamma) if self._n_step > 1 else None

        # Stability
        self._checkpoint_dir = Path("checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._global_step: int = 0
        self._avg_window: list[float] = []
        self._peak_avg: float = 0.0
        self._rollback_count: int = 0
        self._last_rollback_step: int = 0
        self._level_avg_windows: dict[int, list] = {}
        self._level_peak_avgs: dict[int, float] = {}

    # --- Observation helpers ---

    def _obs_to_image(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            return obs["image"]
        return obs

    def _obs_to_features(self, obs) -> np.ndarray | None:
        if isinstance(obs, dict):
            return obs["features"]
        return None

    def _obs_to_storable(self, obs):
        """Convert obs to a format that can be stored in replay buffer."""
        if self._hybrid:
            return (obs["image"].copy(), obs["features"].copy())
        return obs.copy()

    def _stored_to_tensors(self, states_batch):
        """Convert batch of stored states to tensors for network."""
        if self._hybrid:
            images = np.array([s[0] for s in states_batch])
            features = np.array([s[1] for s in states_batch])
            img_t = torch.tensor(images, dtype=torch.uint8).to(self.device).permute(0, 3, 1, 2).float()
            feat_t = torch.tensor(features, dtype=torch.float32).to(self.device)
            return img_t, feat_t
        else:
            images = np.array(states_batch)
            return torch.tensor(images, dtype=torch.uint8).to(self.device).permute(0, 3, 1, 2).float(), None

    def _q_values(self, network, img_t, feat_t=None):
        if self._hybrid:
            return network(img_t, feat_t)
        return network(img_t)

    # --- Core ---

    def _get_epsilon(self, step: int) -> float:
        progress = min(step / self.epsilon_decay, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _select_action(self, obs, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            image = self._obs_to_image(obs)
            img_t = torch.tensor(image, dtype=torch.uint8).unsqueeze(0).to(self.device).permute(0, 3, 1, 2).float()
            feat_t = None
            if self._hybrid:
                features = self._obs_to_features(obs)
                feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self._q_values(self.q_network, img_t, feat_t)
            return int(q_values.argmax(dim=1).item())

    def _train_step(self) -> float:
        result, weights, indices = self.replay_buffer.sample(self.batch_size)
        if result is None:
            return 0.0

        states, actions, rewards, next_states, dones = result

        states_img, states_feat = self._stored_to_tensors(states)
        next_img, next_feat = self._stored_to_tensors(next_states)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        dones_t = torch.tensor(dones).to(self.device)

        # Current Q
        q_values = self._q_values(self.q_network, states_img, states_feat)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q (Double DQN: online network selects action, target network evaluates)
        # N-step: gamma^n for bootstrapping
        gamma_n = self.gamma ** self._n_step if self._n_step > 1 else self.gamma
        with torch.no_grad():
            next_q_online = self._q_values(self.q_network, next_img, next_feat)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self._q_values(self.target_network, next_img, next_feat)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)
            target = rewards_t + gamma_n * next_q * (1 - dones_t)

        # TD error
        td_errors = (q_values - target).detach()

        # Loss
        loss_fn = F.smooth_l1_loss if self._huber else F.mse_loss
        if weights is not None:
            weights_t = torch.tensor(weights).to(self.device)
            loss = (weights_t * loss_fn(q_values, target, reduction='none')).mean()
        else:
            loss = loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self._grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self._grad_clip)
        self.optimizer.step()

        # Reset noise for noisy nets
        if self._noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Update priorities for PER
        if indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

        return loss.item()

    def train(self) -> None:
        obs, info = self.env.reset()
        ep_reward = 0.0
        ep_steps = 0
        start_step = self._global_step
        start_time = time.time()
        loss = 0.0

        for global_step in range(start_step + 1, self.total_timesteps + 1):
            self._global_step = global_step
            # Controls
            if self.controls:
                while self.controls.paused:
                    if self.controls.step_once:
                        self.controls.step_once = False
                        break
                    time.sleep(0.1)
                if self.controls.set_level >= 0:
                    new_level = self.controls.set_level
                    self.controls.set_level = -1
                    old_level = self.env.unwrapped._start_level
                    # Save current level's model + state
                    self._save_level(old_level)
                    self._level_avg_windows[old_level] = list(self._avg_window)
                    self._level_peak_avgs[old_level] = self._peak_avg
                    # Switch level
                    self.env.unwrapped._start_level = new_level
                    # Load new level's model and clear buffer
                    self._load_level(new_level)
                    self._avg_window = list(self._level_avg_windows.get(new_level, []))
                    self._peak_avg = self._level_peak_avgs.get(new_level, 0.0)
                    if self._per:
                        self.replay_buffer = PrioritisedReplayBuffer(
                            self.buffer_size, total_steps=self.total_timesteps, train_freq=self.train_freq)
                    else:
                        self.replay_buffer = ReplayBuffer(self.buffer_size)
                    if self.frame_buffer:
                        self.frame_buffer.current_level = new_level
                        if self.frame_buffer.tracker:
                            self.frame_buffer.tracker.set_current_level(new_level)
                    obs, info = self.env.reset()
                    ep_reward = 0.0
                    ep_steps = 0
                    print(f"Switched to Level {new_level + 1}")
                    continue
                if self.controls.consume_save():
                    level = self.env.unwrapped._start_level
                    self._save_level(level)
                    if self.on_save:
                        self.on_save()
                if self.controls.consume_save_state():
                    # Auto-save model before practice as safety net
                    self.save(str(self._checkpoint_dir / "pre_practice.pt"))
                    self.env.unwrapped.save_game_state()
                    self.controls.practice_mode = True
                    if self.frame_buffer:
                        self.frame_buffer.practice_scroll = self.env.unwrapped._cumulative_scroll
                    print("Model saved to pre_practice.pt — practice mode ON")
                if self.controls.consume_clear_state():
                    self.env.unwrapped.clear_game_state()
                    if self.frame_buffer:
                        self.frame_buffer.practice_scroll = 0
                    print("Practice mode OFF (pre_practice.pt available for rollback)")
                if self.controls.consume_restart():
                    obs, info = self.env.reset()
                    ep_reward = 0.0
                    ep_steps = 0
                    continue
                self.env.unwrapped._max_steps = self.controls.episode_length

            # Select action
            epsilon = self._get_epsilon(global_step)
            action = self._select_action(obs, epsilon)

            # Step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            # Track action usage
            if self.frame_buffer and action < len(self.frame_buffer.action_counts):
                self.frame_buffer.action_counts[action] += 1

            # Update max scroll
            if self.frame_buffer and self.frame_buffer.tracker:
                self.frame_buffer.tracker.update_max_scroll(self.frame_buffer.env0_scroll)

            # Store in replay buffer (with optional N-step)
            if self._nstep_buf:
                self._nstep_buf.push(self._obs_to_storable(obs), action, reward,
                                     self._obs_to_storable(next_obs), done)
                transition = self._nstep_buf.get()
                if transition:
                    self.replay_buffer.push(*transition)
            else:
                self.replay_buffer.push(
                    self._obs_to_storable(obs), action, reward,
                    self._obs_to_storable(next_obs), done
                )

            obs = next_obs

            # Train
            if global_step >= self.learning_starts and global_step % self.train_freq == 0:
                loss = self._train_step()

            # Sync target network
            if global_step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Episode ended
            if done:
                # Flush N-step buffer
                if self._nstep_buf:
                    for transition in self._nstep_buf.flush():
                        self.replay_buffer.push(*transition)

                is_practice = self.env.unwrapped._practice

                # Track practice episodes separately
                if is_practice and self.frame_buffer:
                    self.frame_buffer.practice_rewards.append(ep_reward)
                    if len(self.frame_buffer.practice_rewards) > 500:
                        self.frame_buffer.practice_rewards = self.frame_buffer.practice_rewards[-500:]

                # Only record stats for non-practice episodes
                if not is_practice:
                    if self.on_episode:
                        ep_info = {
                            "step": ep_steps,
                            "deaths": 1 if terminated else 0,
                            "scroll": self.frame_buffer.env0_scroll if self.frame_buffer else 0,
                            "reached_boss": self.env.unwrapped._reached_boss,
                            "reached_boss_level": self.env.unwrapped._reached_boss_level,
                        }
                        self.on_episode(ep_reward, ep_info)

                    # Auto-save stats
                    if self.frame_buffer and self.frame_buffer.tracker:
                        if self.frame_buffer.env0_episode % 50 == 0:
                            self.frame_buffer.tracker.save()

                    # Auto-rollback
                    self._avg_window.append(ep_reward)
                    if len(self._avg_window) > 200:
                        self._avg_window.pop(0)

                    if len(self._avg_window) >= 30:
                        current_avg = np.mean(self._avg_window[-30:])
                        cur_level = self.env.unwrapped._start_level
                        if current_avg > self._peak_avg:
                            self._peak_avg = current_avg
                            self._save_level(cur_level)
                            if self.frame_buffer and self.frame_buffer.tracker:
                                self.frame_buffer.tracker.on_autosave()

                        drop = (self._peak_avg - current_avg) / max(self._peak_avg, 1)
                        if drop > 0.5 and global_step - self._last_rollback_step > 10_000:
                            print(f"\n!!! COLLAPSE DETECTED: avg {current_avg:.0f} vs peak {self._peak_avg:.0f}")
                            print(f"!!! AUTO-ROLLBACK\n")
                            self._load_level(cur_level)
                            self._rollback_count += 1
                            self._last_rollback_step = global_step
                            self._avg_window.clear()
                            if self.frame_buffer and self.frame_buffer.tracker:
                                self.frame_buffer.tracker.on_rollback(self._rollback_count)

                # Frame buffer tracking (always increment for dashboard)
                if self.frame_buffer:
                    self.frame_buffer.env0_episode += 1
                    if not is_practice:
                        self.frame_buffer.on_episode_done(0, ep_reward)

                # Wait if auto-restart is disabled
                if self.controls and not self.controls.auto_restart:
                    self.controls.waiting_restart = True
                    while not self.controls.auto_restart and not self.controls.consume_restart():
                        time.sleep(0.1)
                    self.controls.waiting_restart = False

                obs, info = self.env.reset()
                ep_reward = 0.0
                ep_steps = 0

            # FPS tracking + buffer info
            if global_step % 1000 == 0:
                elapsed = time.time() - start_time
                steps_this_session = global_step - start_step
                fps = steps_this_session / elapsed if elapsed > 0 else 0
                if self.on_step:
                    self.on_step(global_step, fps)
                if self.frame_buffer:
                    self.frame_buffer.buffer_size = len(self.replay_buffer)
                    self.frame_buffer.buffer_capacity = self.buffer_size
                mode = []
                if self._hybrid:
                    mode.append("hybrid")
                if self._per:
                    mode.append("PER")
                mode_str = f" [{'+'.join(mode)}]" if mode else ""
                print(
                    f"Step {global_step:,} | "
                    f"FPS: {fps:.0f} | "
                    f"Loss: {loss:.3f} | "
                    f"Eps: {epsilon:.2f} | "
                    f"Buffer: {len(self.replay_buffer):,}"
                    f"{mode_str}"
                )

    def _save_level(self, level: int) -> None:
        """Save model checkpoint for a specific level."""
        path = self._checkpoint_dir / f"level_{level}.pt"
        self.save(str(path))

    def _load_level(self, level: int) -> None:
        """Load model checkpoint for a specific level. Fresh weights if none exists."""
        path = self._checkpoint_dir / f"level_{level}.pt"
        if path.exists():
            self.load(str(path))
        else:
            # Fresh network for new level
            net_kwargs = {"dueling": self._dueling, "noisy": self._noisy}
            if self._hybrid:
                self.q_network = HybridQNetwork(self.n_actions, **net_kwargs).to(self.device)
                self.target_network = HybridQNetwork(self.n_actions, **net_kwargs).to(self.device)
            else:
                self.q_network = QNetwork(self.n_actions, **net_kwargs).to(self.device)
                self.target_network = QNetwork(self.n_actions, **net_kwargs).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
            print(f"Fresh model for level {level + 1}")

    def save(self, path: str) -> None:
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self._global_step,
            "peak_avg": self._peak_avg,
            "rollback_count": self._rollback_count,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "global_step" in checkpoint:
            self._global_step = checkpoint["global_step"]
        if "peak_avg" in checkpoint:
            self._peak_avg = checkpoint["peak_avg"]
        if "rollback_count" in checkpoint:
            self._rollback_count = checkpoint["rollback_count"]
