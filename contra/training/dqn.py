"""DQN (Deep Q-Network) with replay buffer for Contra NES.

Key difference from PPO: stores experiences and learns from them repeatedly.
Rare events (successful dodge) are seen hundreds of times during training.
"""

from __future__ import annotations

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


class QNetwork(nn.Module):
    """CNN Q-network: maps 84x84x4 observation to Q-value per action."""

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)


class ReplayBuffer:
    """Stores transitions and samples random batches for training."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


class DQNTrainer:
    """DQN training loop with target network and epsilon-greedy exploration."""

    def __init__(
        self,
        env,  # single gymnasium env (not vectorized)
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

        # Hyperparameters
        self.lr = 1e-4
        self.gamma = 0.99
        self.batch_size = 32
        self.buffer_size = 100_000
        self.learning_starts = 1_000  # random exploration before training
        self.train_freq = 4  # train every N steps
        self.target_update_freq = 1_000  # sync target network every N steps
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 50_000  # steps to decay from start to end

        # Networks: online + target (frozen copy for stable Q-targets)
        self.q_network = QNetwork(self.n_actions).to(self.device)
        self.target_network = QNetwork(self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Stability: auto-rollback
        self._checkpoint_dir = Path("checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._avg_window: list[float] = []
        self._peak_avg: float = 0.0
        self._rollback_count: int = 0
        self._last_rollback_step: int = 0

    def _get_epsilon(self, step: int) -> float:
        """Epsilon decays linearly from start to end."""
        progress = min(step / self.epsilon_decay, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy: random action with probability epsilon, best Q otherwise."""
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.uint8).unsqueeze(0).to(self.device)
            t = t.permute(0, 3, 1, 2).float()
            q_values = self.q_network(t)
            return int(q_values.argmax(dim=1).item())

    def _train_step(self) -> float:
        """Sample batch from replay buffer and do one gradient step."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.uint8).to(self.device).permute(0, 3, 1, 2).float()
        next_states_t = torch.tensor(next_states, dtype=torch.uint8).to(self.device).permute(0, 3, 1, 2).float()
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        dones_t = torch.tensor(dones).to(self.device)

        # Current Q values for chosen actions
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (from frozen target network)
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def train(self) -> None:
        """Main training loop."""
        obs, info = self.env.reset()
        ep_reward = 0.0
        ep_steps = 0
        global_step = 0
        start_time = time.time()
        loss = 0.0

        for global_step in range(1, self.total_timesteps + 1):
            # Controls
            if self.controls:
                while self.controls.paused:
                    time.sleep(0.1)
                if self.controls.consume_save() and self.on_save:
                    self.on_save()
                if self.controls.consume_restart():
                    obs, info = self.env.reset()
                    ep_reward = 0.0
                    ep_steps = 0
                    continue
                # Sync episode length
                self.env.unwrapped._max_steps = self.controls.episode_length

            # Select action (epsilon-greedy)
            epsilon = self._get_epsilon(global_step)
            action = self._select_action(obs, epsilon)

            # Step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            # Update max scroll for progress bar
            if self.frame_buffer and self.frame_buffer.tracker:
                self.frame_buffer.tracker.update_max_scroll(self.frame_buffer.env0_scroll)

            # Store in replay buffer
            self.replay_buffer.push(obs, action, reward, next_obs, done)

            obs = next_obs

            # Train from replay buffer
            if global_step >= self.learning_starts and global_step % self.train_freq == 0:
                loss = self._train_step()

            # Sync target network
            if global_step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Episode ended
            if done:
                if self.on_episode:
                    ep_info = {
                        "step": ep_steps,
                        "deaths": 1 if terminated else 0,
                        "scroll": self.frame_buffer.env0_scroll if self.frame_buffer else 0,
                    }
                    self.on_episode(ep_reward, ep_info)

                # Auto-save stats every 50 episodes
                if self.frame_buffer and self.frame_buffer.tracker:
                    if self.frame_buffer.env0_episode % 50 == 0:
                        self.frame_buffer.tracker.save()

                # Auto-rollback tracking
                self._avg_window.append(ep_reward)
                if len(self._avg_window) > 200:
                    self._avg_window.pop(0)

                if len(self._avg_window) >= 30:
                    current_avg = np.mean(self._avg_window[-30:])
                    if current_avg > self._peak_avg:
                        self._peak_avg = current_avg
                        self.save(str(self._checkpoint_dir / "auto_best.pt"))
                        if self.frame_buffer and self.frame_buffer.tracker:
                            self.frame_buffer.tracker.on_autosave()

                    drop = (self._peak_avg - current_avg) / max(self._peak_avg, 1)
                    if drop > 0.5 and global_step - self._last_rollback_step > 10_000:
                        print(f"\n!!! COLLAPSE DETECTED: avg {current_avg:.0f} vs peak {self._peak_avg:.0f}")
                        print(f"!!! AUTO-ROLLBACK\n")
                        best = self._checkpoint_dir / "auto_best.pt"
                        if best.exists():
                            self.load(str(best))
                            self._rollback_count += 1
                            self._last_rollback_step = global_step
                            self._avg_window.clear()
                            if self.frame_buffer and self.frame_buffer.tracker:
                                self.frame_buffer.tracker.on_rollback(self._rollback_count)

                # Frame buffer tracking
                if self.frame_buffer:
                    self.frame_buffer.env0_episode += 1
                    self.frame_buffer.on_episode_done(0, ep_reward)

                obs, info = self.env.reset()
                ep_reward = 0.0
                ep_steps = 0

            # FPS tracking
            if global_step % 1000 == 0:
                elapsed = time.time() - start_time
                fps = global_step / elapsed if elapsed > 0 else 0
                if self.on_step:
                    self.on_step(global_step, fps)
                print(
                    f"Step {global_step:,} | "
                    f"FPS: {fps:.0f} | "
                    f"Loss: {loss:.3f} | "
                    f"Eps: {epsilon:.2f} | "
                    f"Buffer: {len(self.replay_buffer):,}"
                )

    def save(self, path: str) -> None:
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
