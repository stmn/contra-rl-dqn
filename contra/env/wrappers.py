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
        features = self.env.unwrapped._build_features() if hasattr(self.env.unwrapped, '_build_features') else None
        self._fb.write_env(self._env_id, frame, info.get("scroll", 0), raw_frame=raw, features=features)
        # Update run log for dashboard Logs tab
        if self._env_id == 0:
            nes = self.env.unwrapped._nes
            from contra.env.contra_env import RAM_WEAPON
            px, py = int(nes[0x334]), int(nes[0x31A])
            weapon = int(nes[RAM_WEAPON] & 0x07)

            # Build visible entities list (same addresses as overlay)
            _BULLET_TYPES = {0x01, 0x0B, 0x0F}  # enemy bullet, mortar, turret bullet
            _OTHER_TYPES = {0x02, 0x03, 0x12}    # weapon box, flying bonus, bridge
            visible_enemies = []
            visible_bullets = []
            visible_other = []
            for slot in range(16):
                etype = nes[0x528 + slot]
                ehp = nes[0x578 + slot]
                routine = nes[0x4B8 + slot]
                if etype == 0 and ehp == 0:
                    continue
                if routine == 0 or ehp == 0:
                    continue
                ex, ey = int(nes[0x33E + slot]), int(nes[0x324 + slot])
                if ey > 230 or ey < 8 or ex < 24 or ex > 240:
                    continue
                turret_hp = int(nes[0x578 + slot])
                dist = abs(ex - px) + abs(ey - py)
                if etype in _BULLET_TYPES:
                    visible_bullets.append({"x": ex, "y": ey, "dist": dist})
                elif etype in _OTHER_TYPES:
                    entry = {"x": ex, "y": ey, "dist": dist, "type": int(etype)}
                    if etype in (0x00, 0x02, 0x03):  # weapon items/boxes
                        entry["weapon"] = int(nes[0x5A8 + slot]) & 0x07
                    visible_other.append(entry)
                else:
                    # Only turrets/bosses have real HP at $580: types 4,7,8,0xE,0x10,0x11
                    _HP_TYPES = {0x04, 0x07, 0x08, 0x0E, 0x10, 0x11}
                    hp = turret_hp if etype in _HP_TYPES and 0 < turret_hp <= 32 else 1
                    visible_enemies.append({"x": ex, "y": ey, "dist": dist, "type": int(etype), "hp": hp})

            visible_enemies.sort(key=lambda e: e["dist"])
            visible_bullets.sort(key=lambda b: b["dist"])

            self._fb.env0_run_log = {
                "step": info.get("step", 0),
                "total_reward": round(info.get("total_reward", 0), 1),
                "scroll": info.get("scroll", 0),
                "deaths": info.get("deaths", 0),
                "reward_scroll": info.get("reward_scroll", 0),
                "reward_death": info.get("reward_death", 0),
                "reward_kills": info.get("reward_kills", 0),
                "reward_turret": info.get("reward_turret", 0),
                "reward_idle": info.get("reward_idle", 0),
                "reward_weapon": info.get("reward_weapon", 0),
                "turret_hits": info.get("turret_hits", 0),
                "events": info.get("events", []),
                "player": {"x": px, "y": py, "weapon": weapon},
                "enemies": visible_enemies[:5],
                "bullets": visible_bullets[:5],
                "other": visible_other[:5],
            }
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw = self.env.unwrapped._raw_frame
        frame = _get_image(obs)
        features = obs.get("features") if isinstance(obs, dict) else None
        self._fb.write_env(self._env_id, frame, info.get("scroll", 0), raw_frame=raw, features=features)
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
