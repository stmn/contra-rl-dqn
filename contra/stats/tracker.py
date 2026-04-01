"""Thread-safe training statistics tracker with persistence."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class StatsSnapshot:
    """Immutable snapshot of current stats for overlay/web rendering."""

    episode: int = 0
    current_reward: float = 0.0
    best_reward: float = 0.0
    best_level: int = 0
    deaths_this_run: int = 0
    total_deaths: int = 0
    training_start: float = 0.0  # time.time()
    timesteps: int = 0
    total_timesteps: int = 100_000_000
    fps: float = 0.0
    generation: int = 0  # full game-overs
    # Reward breakdown (last episode)
    last_reward_scroll: float = 0.0
    last_reward_death: float = 0.0
    last_death_count: int = 0
    timeout_pct: float = 0.0
    rollback_count: int = 0
    last_rollback_ago: float = 0.0
    last_autosave_ago: float = 0.0


@dataclass
class EpisodeRecord:
    """Record of a completed episode for top runs."""

    episode: int
    reward: float
    level: int
    deaths: int
    timestamp: float
    steps: int = 0


class StatsTracker:
    """Thread-safe stats tracking with periodic persistence to disk."""

    def __init__(self, save_path: Path, total_timesteps: int = 100_000_000) -> None:
        self._lock = threading.Lock()
        self._save_path = save_path
        self._total_timesteps = total_timesteps

        # Current state
        self._episode = 0
        self._current_reward = 0.0
        self._best_reward = 0.0
        self._best_level = 0
        self._deaths_this_run = 0
        self._total_deaths = 0
        self._training_start = time.time()
        self._timesteps = 0
        self._fps = 0.0
        self._generation = 0
        self._last_reward_scroll = 0.0
        self._last_reward_death = 0.0
        self._last_death_count = 0
        self._best_reward_time: float = 0.0  # timestamp of last PB
        self._death_positions: list[int] = []  # scroll positions where agent died
        self._max_scroll_seen: int = 0

        # History for charts
        self._reward_history: list[float] = []
        self._survival_history: list[int] = []
        self._boss_history: list[bool] = []  # True = reached boss
        self._recent_timeouts: list[bool] = []  # last 50 episodes: True = hit limit
        self._episode_length: int = 18_000  # 10 min safety net
        self._rollback_count: int = 0
        self._last_rollback_time: float = 0.0
        self._last_autosave_time: float = 0.0
        self._top_runs: list[EpisodeRecord] = []

        self.load()

    def on_episode_end(self, reward: float, level: int, deaths: int, info: dict | None = None) -> None:
        with self._lock:
            self._episode += 1
            self._current_reward = reward
            self._deaths_this_run = deaths
            self._total_deaths += deaths
            self._generation += 1

            if reward > self._best_reward:
                self._best_reward = reward
                self._best_reward_time = time.time()
            if level > self._best_level:
                self._best_level = level

            if info:
                self._last_reward_scroll = info.get("reward_scroll", 0.0)
                self._last_reward_death = info.get("reward_death", 0.0)
                self._last_death_count = info.get("death_count", 0)
                scroll = int(info.get("scroll", 0))
                if scroll > self._max_scroll_seen:
                    self._max_scroll_seen = scroll
                if deaths > 0:
                    self._death_positions.append(scroll)
                    if len(self._death_positions) > 5000:
                        self._death_positions = self._death_positions[-2500:]

            ep_steps = int(info.get("step", 0)) if info else 0
            hit_limit = info.get("deaths", 0) == 0 if info else False  # no death = timeout
            self._recent_timeouts.append(hit_limit)
            if len(self._recent_timeouts) > 50:
                self._recent_timeouts.pop(0)
            reached_boss = info.get("reached_boss", False) if info else False
            self._reward_history.append(reward)
            self._survival_history.append(ep_steps)
            self._boss_history.append(reached_boss)
            if len(self._reward_history) > 10_000:
                self._reward_history = self._reward_history[-5_000:]
                self._survival_history = self._survival_history[-5_000:]
                self._boss_history = self._boss_history[-5_000:]

            record = EpisodeRecord(
                episode=self._episode,
                reward=reward,
                level=level,
                deaths=deaths,
                timestamp=time.time(),
                steps=ep_steps,
            )
            self._top_runs.append(record)
            self._top_runs.sort(key=lambda r: r.reward, reverse=True)
            self._top_runs = self._top_runs[:20]

    def update_step(self, timesteps: int, fps: float) -> None:
        with self._lock:
            self._timesteps = timesteps
            self._fps = fps

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            return StatsSnapshot(
                episode=self._episode,
                current_reward=self._current_reward,
                best_reward=self._best_reward,
                best_level=self._best_level,
                deaths_this_run=self._deaths_this_run,
                total_deaths=self._total_deaths,
                training_start=self._training_start,
                timesteps=self._timesteps,
                total_timesteps=self._total_timesteps,
                fps=self._fps,
                generation=self._generation,
                last_reward_scroll=self._last_reward_scroll,
                last_reward_death=self._last_reward_death,
                last_death_count=self._last_death_count,
                timeout_pct=round(sum(self._recent_timeouts) / max(len(self._recent_timeouts), 1) * 100),
                rollback_count=self._rollback_count,
                last_rollback_ago=round(time.time() - self._last_rollback_time) if self._last_rollback_time > 0 else 0,
                last_autosave_ago=round(time.time() - self._last_autosave_time) if self._last_autosave_time > 0 else 0,
            )

    def death_positions(self) -> list[int]:
        with self._lock:
            return self._death_positions.copy()

    def time_since_pb(self) -> float:
        with self._lock:
            if self._best_reward_time == 0:
                return 0.0
            return time.time() - self._best_reward_time

    def on_rollback(self, count: int) -> None:
        with self._lock:
            self._rollback_count = count
            self._last_rollback_time = time.time()

    def on_autosave(self) -> None:
        with self._lock:
            self._last_autosave_time = time.time()

    def update_max_scroll(self, scroll: int) -> None:
        with self._lock:
            if scroll > self._max_scroll_seen:
                self._max_scroll_seen = scroll

    def max_scroll(self) -> int:
        with self._lock:
            return self._max_scroll_seen

    def reward_history(self, last_n: int = 1000) -> list[float]:
        with self._lock:
            return self._reward_history[-last_n:]

    def survival_history(self, last_n: int = 1000) -> list[int]:
        with self._lock:
            return self._survival_history[-last_n:]

    def boss_history(self, last_n: int = 1000) -> list[bool]:
        with self._lock:
            return self._boss_history[-last_n:]

    def top_runs(self, n: int = 10) -> list[dict]:
        with self._lock:
            return [
                {"episode": r.episode, "reward": r.reward, "level": r.level, "duration": round(r.steps / 15, 1)}
                for r in self._top_runs[:n]
            ]

    def save(self) -> None:
        with self._lock:
            data = {
                "episode": self._episode,
                "best_reward": self._best_reward,
                "best_level": self._best_level,
                "total_deaths": self._total_deaths,
                "training_start": self._training_start,
                "timesteps": self._timesteps,
                "generation": self._generation,
                "reward_history": self._reward_history[-5_000:],
                "survival_history": self._survival_history[-5_000:],
                "boss_history": self._boss_history[-5_000:],
                "episode_length": self._episode_length,
                "top_runs": [
                    {"episode": r.episode, "reward": r.reward, "level": r.level, "deaths": r.deaths, "timestamp": r.timestamp, "steps": r.steps}
                    for r in self._top_runs
                ],
            }
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        if not self._save_path.exists():
            return
        try:
            data = json.loads(self._save_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        with self._lock:
            self._episode = data.get("episode", 0)
            self._best_reward = data.get("best_reward", 0.0)
            self._best_level = data.get("best_level", 0)
            self._total_deaths = data.get("total_deaths", 0)
            self._training_start = data.get("training_start", self._training_start)
            self._timesteps = data.get("timesteps", 0)
            self._generation = data.get("generation", 0)
            self._reward_history = data.get("reward_history", [])
            self._survival_history = data.get("survival_history", [])
            self._boss_history = data.get("boss_history", [])
            self._episode_length = data.get("episode_length", 600)
            self._top_runs = [
                EpisodeRecord(
                    episode=r["episode"], reward=r["reward"],
                    level=r["level"], deaths=r["deaths"],
                    timestamp=r.get("timestamp", 0),
                    steps=r.get("steps", 0),
                )
                for r in data.get("top_runs", [])
            ]
