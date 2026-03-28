"""Preprocessing wrappers for Contra NES environment."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StreamCapture(gym.Wrapper):
    """Captures raw RGB frames and sends them to frame buffer."""

    def __init__(self, env: gym.Env, frame_buffer, env_id: int) -> None:
        super().__init__(env)
        self._fb = frame_buffer
        self._env_id = env_id

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs has overlay; raw_frame is without overlay (for dashboard preview)
        raw = self.env.unwrapped._raw_frame
        self._fb.write_env(self._env_id, obs, info.get("scroll", 0), raw_frame=raw)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw = self.env.unwrapped._raw_frame
        self._fb.write_env(self._env_id, obs, info.get("scroll", 0), raw_frame=raw)
        return obs, info


class Grayscale(gym.ObservationWrapper):
    """Convert RGB observation to grayscale."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        h, w = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]


class Resize(gym.ObservationWrapper):
    """Resize observation to target shape."""

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)) -> None:
        super().__init__(env)
        self._shape = shape
        channels = self.observation_space.shape[2] if len(self.observation_space.shape) > 2 else 1
        self.observation_space = spaces.Box(low=0, high=255, shape=(*shape, channels), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        resized = cv2.resize(obs, self._shape[::-1], interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return resized


class FrameStack(gym.Wrapper):
    """Stack N consecutive frames as channels for temporal awareness."""

    def __init__(self, env: gym.Env, n_stack: int = 4) -> None:
        super().__init__(env)
        self._n_stack = n_stack
        h, w, c = env.observation_space.shape
        self._frames = np.zeros((h, w, c * n_stack), dtype=np.uint8)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c * n_stack), dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames[:] = 0
        self._frames[:, :, -obs.shape[2]:] = obs
        return self._frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames[:, :, :-obs.shape[2]] = self._frames[:, :, obs.shape[2]:]
        self._frames[:, :, -obs.shape[2]:] = obs
        return self._frames.copy(), reward, terminated, truncated, info


def wrap_contra(
    env: gym.Env,
    frame_buffer=None,
    env_id: int = 0,
) -> gym.Env:
    """Apply the full preprocessing pipeline."""
    if frame_buffer:
        env = StreamCapture(env, frame_buffer=frame_buffer, env_id=env_id)
    env = Grayscale(env)
    env = Resize(env, shape=(128, 128))
    env = FrameStack(env, n_stack=4)
    return env
