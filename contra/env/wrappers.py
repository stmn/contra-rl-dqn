"""Preprocessing wrappers for Contra NES environment."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.settings import settings


def _get_image(obs):
    """Extract image from obs (dict or raw array)."""
    return obs["image"] if isinstance(obs, dict) else obs


def _set_image(obs, image):
    """Return obs with replaced image."""
    if isinstance(obs, dict):
        return {**obs, "image": image}
    return image


class StreamCapture(gym.Wrapper):
    """Captures raw RGB frames and sends them to frame buffer."""

    def __init__(self, env: gym.Env, frame_buffer, env_id: int) -> None:
        super().__init__(env)
        self._fb = frame_buffer
        self._env_id = env_id

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped._raw_frame
        frame = _get_image(obs)
        self._fb.write_env(self._env_id, frame, info.get("scroll", 0), raw_frame=raw)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw = self.env.unwrapped._raw_frame
        frame = _get_image(obs)
        self._fb.write_env(self._env_id, frame, info.get("scroll", 0), raw_frame=raw)
        return obs, info


class Grayscale(gym.ObservationWrapper):
    """Convert RGB observation to grayscale. Handles dict obs (hybrid mode)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if isinstance(self.observation_space, spaces.Dict):
            h, w = self.observation_space["image"].shape[:2]
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8),
                "features": self.observation_space["features"],
            })
        else:
            h, w = self.observation_space.shape[:2]
            self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)

    def observation(self, obs):
        image = _get_image(obs)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        return _set_image(obs, gray)


class Resize(gym.ObservationWrapper):
    """Resize observation to target shape. Handles dict obs."""

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)) -> None:
        super().__init__(env)
        self._shape = shape
        if isinstance(self.observation_space, spaces.Dict):
            channels = self.observation_space["image"].shape[2]
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(*shape, channels), dtype=np.uint8),
                "features": self.observation_space["features"],
            })
        else:
            channels = self.observation_space.shape[2] if len(self.observation_space.shape) > 2 else 1
            self.observation_space = spaces.Box(low=0, high=255, shape=(*shape, channels), dtype=np.uint8)

    def observation(self, obs):
        image = _get_image(obs)
        resized = cv2.resize(image, self._shape[::-1], interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return _set_image(obs, resized)


class FrameStack(gym.Wrapper):
    """Stack N consecutive frames as channels. Handles dict obs (features pass through)."""

    def __init__(self, env: gym.Env, n_stack: int = 4) -> None:
        super().__init__(env)
        self._n_stack = n_stack
        self._hybrid = isinstance(env.observation_space, spaces.Dict)

        if self._hybrid:
            h, w, c = env.observation_space["image"].shape
            self._frames = np.zeros((h, w, c * n_stack), dtype=np.uint8)
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(h, w, c * n_stack), dtype=np.uint8),
                "features": env.observation_space["features"],
            })
        else:
            h, w, c = env.observation_space.shape
            self._frames = np.zeros((h, w, c * n_stack), dtype=np.uint8)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, c * n_stack), dtype=np.uint8,
            )

    def _stack(self, image):
        self._frames[:, :, :-image.shape[2]] = self._frames[:, :, image.shape[2]:]
        self._frames[:, :, -image.shape[2]:] = image
        return self._frames.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames[:] = 0
        image = _get_image(obs)
        stacked = self._stack(image)
        if self._hybrid:
            return {"image": stacked, "features": obs["features"]}, info
        return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        image = _get_image(obs)
        stacked = self._stack(image)
        if self._hybrid:
            return {"image": stacked, "features": obs["features"]}, reward, terminated, truncated, info
        return stacked, reward, terminated, truncated, info


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
