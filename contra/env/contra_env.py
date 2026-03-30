"""Gymnasium-compatible Contra NES environment using cynes (Rust NES emulator)."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.settings import settings

# NES controller button mapping — cynes uses reversed bit order!
import cynes as _cynes
BUTTON_A = _cynes.NES_INPUT_A          # 128
BUTTON_B = _cynes.NES_INPUT_B          # 64
BUTTON_SELECT = _cynes.NES_INPUT_SELECT  # 32
BUTTON_START = _cynes.NES_INPUT_START  # 16
BUTTON_UP = _cynes.NES_INPUT_UP        # 8
BUTTON_DOWN = _cynes.NES_INPUT_DOWN    # 4
BUTTON_LEFT = _cynes.NES_INPUT_LEFT    # 2
BUTTON_RIGHT = _cynes.NES_INPUT_RIGHT  # 1

# Action table: meaningful button combos for Contra
ACTIONS = [
    0,                                       # 0: NOOP
    BUTTON_RIGHT,                            # 1: Right
    BUTTON_LEFT,                             # 2: Left
    BUTTON_UP,                               # 3: Up (aim up)
    BUTTON_DOWN,                             # 4: Down (crouch/aim down)
    BUTTON_A,                                # 5: Jump
    BUTTON_B,                                # 6: Shoot
    BUTTON_RIGHT | BUTTON_A,                 # 7: Right + Jump
    BUTTON_RIGHT | BUTTON_B,                 # 8: Right + Shoot
    BUTTON_RIGHT | BUTTON_A | BUTTON_B,      # 9: Right + Jump + Shoot
    BUTTON_LEFT | BUTTON_A,                  # 10: Left + Jump
    BUTTON_LEFT | BUTTON_B,                  # 11: Left + Shoot
    BUTTON_UP | BUTTON_B,                    # 12: Up + Shoot (aim up + fire)
    BUTTON_DOWN | BUTTON_B,                  # 13: Down + Shoot (crouch + fire)
    BUTTON_RIGHT | BUTTON_UP | BUTTON_B,     # 14: Right + Up + Shoot (diagonal fire)
    BUTTON_RIGHT | BUTTON_DOWN | BUTTON_B,   # 15: Right + Down + Shoot
]

# RAM addresses for Contra (NES) — verified with cynes
RAM_PLAYER_STATE = 0x0090  # 0=falling/respawn, 1=alive, 2=dead
RAM_LEVEL = 0x0030         # 0-7 = stage 1-8
RAM_PLAYER_MODE = 0x0022   # 0=1P, 1=2P
RAM_SCROLL_HI = 0x0060     # camera scroll coarse (tiles, wraps 0-255)
RAM_SCROLL_LO = 0x0065     # camera scroll fine (pixels within tile)
RAM_SCORE = 0x07E2         # Player 1 score (2 bytes, low byte)

# Action categories for reward shaping
_RIGHT_ACTIONS = frozenset(i for i, a in enumerate(ACTIONS) if a & BUTTON_RIGHT)
_LEFT_ACTIONS = frozenset(i for i, a in enumerate(ACTIONS) if a & BUTTON_LEFT)
_SHOOT_ACTIONS = frozenset(i for i, a in enumerate(ACTIONS) if a & BUTTON_B)


class ContraEnv(gym.Env):
    """Gymnasium environment wrapping cynes NES emulator for Contra."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | Path | None = None,
        render_mode: str | None = "rgb_array",
        frame_skip: int = 2,
        max_episode_steps: int = 18_000,  # 10 min safety net, game over ends sooner
        overlay_sprites: bool = False,
    ) -> None:
        super().__init__()
        from cynes import NES

        self._rom_path = str(rom_path or settings.rom_path)
        self._nes = NES(self._rom_path)
        self._frame_skip = frame_skip
        self._max_steps = max_episode_steps
        self._overlay_sprites = overlay_sprites
        self._render_mode = render_mode
        self._hybrid_obs = settings.hybrid_observation

        # Gymnasium spaces
        self.action_space = spaces.Discrete(len(ACTIONS))
        if self._hybrid_obs:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8),
                "features": spaces.Box(low=0.0, high=1.0, shape=(28,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(240, 256, 3), dtype=np.uint8,
            )

        # State tracking
        self._step_count = 0
        self._deaths = 0
        self._total_reward = 0.0
        self._prev_scroll = 0
        self._raw_scroll_prev = 0
        self._cumulative_scroll = 0
        self._last_frame: np.ndarray | None = None
        self._raw_frame: np.ndarray | None = None
        self._force_restart = False
        self._idle_counter = 0
        self._death_this_frame = False
        self._prev_score = 0
        self._saved_game_state: np.ndarray | None = None
        self._practice = False

        # Reward breakdown counters
        self._reward_scroll = 0.0
        self._reward_death = 0.0
        self._death_count = 0

        # Save initial state after selecting 1 player
        self._initial_state: np.ndarray | None = None
        self._boot()

    def _boot(self) -> None:
        """Advance past title screen, select 1 PLAYER, save initial state."""
        self._nes.reset()

        # Wait for title screen to fully load (needs ~300 frames)
        for _ in range(300):
            self._nes.step(1)

        # Press Start → go to player select menu
        self._nes.controller = BUTTON_START
        for _ in range(5):
            self._nes.step(1)
        self._nes.controller = 0

        # Wait for menu to register
        for _ in range(60):
            self._nes.step(1)

        # Press Up to ensure cursor is on "1 PLAYER"
        self._nes.controller = BUTTON_UP
        for _ in range(5):
            self._nes.step(1)
        self._nes.controller = 0
        for _ in range(15):
            self._nes.step(1)

        # Press Start → confirm 1 PLAYER
        self._nes.controller = BUTTON_START
        for _ in range(5):
            self._nes.step(1)
        self._nes.controller = 0

        # Wait for game to fully load
        for _ in range(600):
            self._nes.step(1)

        self._initial_state = self._nes.save()  # numpy array

    def request_restart(self) -> None:
        """Called from web UI — forces episode to end on next step."""
        self._force_restart = True

    def save_game_state(self) -> None:
        """Save current NES state. Future resets will load from here (practice mode)."""
        self._saved_game_state = self._nes.save()
        self._practice = True

    def clear_game_state(self) -> None:
        """Remove saved NES state. Resets go back to level start."""
        self._saved_game_state = None
        self._practice = False

    def _read_raw_scroll(self) -> int:
        return self._nes[RAM_SCROLL_HI] * 256 + self._nes[RAM_SCROLL_LO]

    def _read_scroll(self) -> int:
        """Cumulative scroll that handles wraparound."""
        raw = self._read_raw_scroll()
        delta = raw - self._raw_scroll_prev
        # Detect wraparound: $60 wraps 255→0 = jump of ~-65000
        if delta < -30000:
            delta += 65536
        elif delta > 30000:
            delta = 0  # glitch, ignore
        if delta > 0:
            self._cumulative_scroll += delta
        self._raw_scroll_prev = raw
        return self._cumulative_scroll

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._saved_game_state is not None:
            self._nes.load(self._saved_game_state)
        elif self._initial_state is not None:
            self._nes.load(self._initial_state)
        else:
            self._boot()

        self._step_count = 0
        self._deaths = 0
        self._total_reward = 0.0

        self._raw_scroll_prev = self._read_raw_scroll()
        self._cumulative_scroll = 0
        self._prev_scroll = 0
        self._reward_scroll = 0.0
        self._idle_counter = 0
        self._death_this_frame = False
        self._prev_score = self._nes[RAM_SCORE] + self._nes[RAM_SCORE + 1] * 256
        self._reward_death = 0.0
        self._death_count = 0
        frame = self._nes.step(1)
        self._last_frame = frame.copy()
        info = self._get_info()
        obs = self._make_obs(frame)
        return obs, info

    def _make_obs(self, frame):
        if self._hybrid_obs:
            return {"image": frame, "features": self._build_features()}
        return frame

    def step(self, action: int):
        # Handle forced restart from web UI — just signal termination,
        # let SyncVectorEnv handle the actual reset
        if self._force_restart:
            self._force_restart = False
            info = self._get_info()
            obs = self._make_obs(self._last_frame)
            return obs, 0.0, True, False, info

        controller_input = ACTIONS[action]
        total_reward = 0.0
        terminated = False
        frame = None

        # Frame skip with max-pool of last 2 frames
        frames_buf = [None, None]
        for i in range(self._frame_skip):
            self._nes.controller = controller_input
            frame = self._nes.step(1)
            if i >= self._frame_skip - 2:
                frames_buf[i - (self._frame_skip - 2)] = frame.copy()

            # Death: penalty but continue playing (3 lives)
            player_state = self._nes[RAM_PLAYER_STATE]
            if player_state == 2 and not self._death_this_frame:
                self._deaths += 1
                self._death_count += 1
                self._reward_death += settings.death_penalty
                total_reward += settings.death_penalty
                self._death_this_frame = True
            elif player_state != 2:
                self._death_this_frame = False

            # Game over (all lives lost) or crash = episode ends
            if self._nes[0x38] == 1 or self._nes.has_crashed:
                terminated = True
                break

        # Max of last 2 frames (reduce flickering)
        if frames_buf[0] is not None and frames_buf[1] is not None:
            frame = np.maximum(frames_buf[0], frames_buf[1])
        elif frame is None:
            frame = self._last_frame

        # Save raw frame before overlay (for dashboard preview)
        self._raw_frame = frame.copy() if frame is not None else None

        # Apply sprite overlay if enabled (agent sees this)
        frame = self._apply_overlay(frame)

        self._step_count += 1

        # --- Reward shaping ---
        # Based on ACTUAL scroll progress (camera moved right)

        scroll = self._read_scroll()
        scroll_delta = scroll - self._prev_scroll
        # Handle wraparound (256*255 → 0)
        if scroll_delta < -1000:
            scroll_delta += 256 * 256
        elif scroll_delta > 1000:
            scroll_delta = 0  # ignore glitches

        # Speed bonus: faster scroll = more reward per pixel
        speed_multiplier = 1.0 + min(scroll_delta / 100.0, 1.0)
        scroll_reward = scroll_delta * 0.08 * speed_multiplier
        total_reward += scroll_reward
        self._reward_scroll += scroll_reward
        self._prev_scroll = scroll

        # Kill reward — score increased = enemy killed
        score = self._nes[RAM_SCORE] + self._nes[RAM_SCORE + 1] * 256
        score_delta = score - self._prev_score
        if 0 < score_delta < 100:  # sanity check
            total_reward += score_delta * 15.0
        self._prev_score = score

        # Idle penalty: standing still = negative reward (skip during death/respawn)
        if scroll_delta == 0 and self._nes[RAM_PLAYER_STATE] == 1:
            self._idle_counter += 1
            if self._idle_counter > 10:  # grace period ~0.7s
                total_reward -= 0.5
            # Penalize LEFT at left screen edge (player X < 20px)
            if action in _LEFT_ACTIONS and self._nes[0x334] < 30:
                total_reward -= 0.3
        else:
            self._idle_counter = 0

        # (Death penalty is applied above in frame skip loop)

        self._last_frame = frame.copy() if frame is not None else None

        # Clamp reward
        total_reward = float(np.clip(total_reward, -50.0, 50.0))
        self._total_reward += total_reward

        # Truncation: time limit (death ends episode via terminated)
        truncated = self._step_count >= self._max_steps

        info = self._get_info()
        obs = self._make_obs(frame)
        return obs, total_reward, terminated, truncated, info

    def _get_info(self) -> dict:
        return {
            "step": self._step_count,
            "total_reward": self._total_reward,
            "deaths": self._deaths,
            "level": self._nes[RAM_LEVEL],
            "scroll": self._read_scroll(),
            "player_state": self._nes[RAM_PLAYER_STATE],
            "reward_scroll": round(self._reward_scroll, 1),
            "reward_death": round(self._reward_death, 1),
            "death_count": self._death_count,
            "practice": self._practice,
        }

    def _build_features(self) -> np.ndarray:
        """Build 28-dim feature vector from RAM for hybrid observation."""
        nes = self._nes
        f = np.zeros(28, dtype=np.float32)

        # Player (4 features)
        px = nes[0x334]
        py = nes[0x31A]
        f[0] = px / 256.0
        f[1] = py / 240.0
        f[2] = 1.0 if nes[RAM_PLAYER_STATE] == 1 else 0.0
        f[3] = 1.0 if (nes[0xA0] & 0x0F) > 0 else 0.0  # in air

        # Find nearest 3 enemies by distance to player
        enemies = []
        for slot in range(16):
            etype = nes[0x528 + slot]
            ehp = nes[0x578 + slot]
            if etype == 0 and ehp == 0:
                continue
            ex = nes[0x33E + slot]
            ey = nes[0x324 + slot]
            if ey > 230 or ey < 8 or ex < 8 or ex > 248:
                continue
            dist = abs(ex - px) + abs(ey - py)
            enemies.append((dist, ex, ey, slot))

        enemies.sort(key=lambda e: e[0])

        # Nearest 3 enemies (4 features each: dx, dy, velocity, attack delay)
        for i, (dist, ex, ey, slot) in enumerate(enemies[:3]):
            base = 4 + i * 4
            f[base] = np.clip((ex - px + 128) / 256.0, 0.0, 1.0)
            f[base + 1] = np.clip((ey - py + 120) / 240.0, 0.0, 1.0)
            f[base + 2] = nes[0x508 + slot] / 256.0
            f[base + 3] = nes[0x558 + slot] / 256.0

        # Nearest 3 projectiles (score_code == 0, 2 features each: dx, dy)
        projectiles = []
        for slot in range(16):
            etype = nes[0x528 + slot]
            ehp = nes[0x578 + slot]
            if etype == 0 and ehp == 0:
                continue
            score_code = nes[0x588 + slot] >> 4
            if score_code != 0:
                continue  # not a projectile
            ex = nes[0x33E + slot]
            ey = nes[0x324 + slot]
            if ey > 230 or ey < 8 or ex < 8 or ex > 248:
                continue
            dist = abs(ex - px) + abs(ey - py)
            projectiles.append((dist, ex, ey))

        projectiles.sort(key=lambda p: p[0])

        for i, (dist, bx, by) in enumerate(projectiles[:3]):
            base = 16 + i * 2
            f[base] = np.clip((bx - px + 128) / 256.0, 0.0, 1.0)
            f[base + 1] = np.clip((by - py + 120) / 240.0, 0.0, 1.0)

        # Aggregate (3 features)
        f[22] = min(len(enemies), 16) / 16.0
        f[23] = np.clip(enemies[0][0] / 400.0, 0.0, 1.0) if enemies else 1.0
        f[24] = 1.0 if nes[0x8E] > 0 else 0.0

        # Player movement (3 features)
        f[25] = min((nes[0xA4] & 0x60) / 96.0, 1.0)
        f[26] = 1.0 if (nes[0xA0] & 0x0F) > 0 else 0.0
        y_vel = nes[0xC6]
        f[27] = np.clip((y_vel if y_vel < 128 else y_vel - 256) / 128.0 + 0.5, 0.0, 1.0)

        return f

    # Overlay config: type → (shade, shape)
    # Shapes: 'circle', 'vrect' (vertical rect), 'triangle', 'diamond'
    _OVERLAY_CONFIG = {
        0x01: (255, 'circle'),     # enemy bullet — small bright circle
        0x02: (180, 'diamond'),    # destructible box (bonus) — diamond
        0x03: (180, 'diamond'),    # flying bonus — diamond
        0x04: (255, 'triangle'),   # static turret — triangle
        0x05: (255, 'vrect'),      # running soldier — vertical rect
        0x06: (255, 'vrect'),      # static enemy/sniper — vertical rect
    }

    def _apply_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw sprite overlays: enemies, bullets, player with type labels."""
        if not self._overlay_sprites or frame is None:
            return frame
        import cv2
        f = frame.copy()

        for slot in range(16):
            etype = self._nes[0x528 + slot]
            ehp = self._nes[0x578 + slot]
            routine = self._nes[0x4B8 + slot]
            if etype == 0 and ehp == 0:
                continue
            if routine == 0 or ehp == 0:
                continue  # dead/dying — skip explosion animation
            ex = self._nes[0x33E + slot]
            ey = self._nes[0x324 + slot]
            if ey > 230 or ey < 8 or ex < 24 or ex > 240:
                continue

            config = self._OVERLAY_CONFIG.get(etype)
            if config is None:
                continue

            shade, shape = config
            color = (shade, shade, shade)

            if shape == 'circle':
                cv2.circle(f, (ex, ey), 3, color, -1)
            elif shape == 'vrect':
                cv2.rectangle(f, (max(0, ex - 5), max(0, ey - 10)),
                              (min(255, ex + 5), min(239, ey + 10)), color, -1)
            elif shape == 'triangle':
                pts = np.array([[ex, ey - 10], [ex - 8, ey + 6], [ex + 8, ey + 6]], dtype=np.int32)
                cv2.fillPoly(f, [pts], color)
            elif shape == 'diamond':
                pts = np.array([[ex, ey - 8], [ex + 6, ey], [ex, ey + 8], [ex - 6, ey]], dtype=np.int32)
                cv2.fillPoly(f, [pts], color)

        # Player — medium gray box
        px, py = self._nes[0x334], self._nes[0x31A]
        if 0 < px < 255 and 0 < py < 230:
            cv2.rectangle(f, (px - 6, py - 12), (px + 6, py + 4), (180, 180, 180), -1)

        # Player bullets
        for b in range(16):
            bslot = self._nes[0x388 + b]
            if bslot > 0:
                bx, by = self._nes[0x3C8 + b], self._nes[0x3B8 + b]
                if 0 < bx < 255 and 0 < by < 240:
                    cv2.circle(f, (bx, by), 3, (200, 200, 200), -1)

        return f

    def render(self):
        return self._last_frame

    def close(self):
        pass
