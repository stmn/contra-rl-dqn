"""Custom callbacks for frame capture and stats tracking."""

from __future__ import annotations

import threading
import time

import numpy as np

from contra.stats.tracker import StatsTracker


class StatsCallback:
    pass  # unused, kept for import compat


class SharedFrameBuffer:
    """Thread-safe frame buffer with per-env slots.

    Each env writes its own frame+scroll. The active env determines
    which frame is displayed. Switching active env instantly shows
    the correct frame — no timing gaps.
    """

    def __init__(self, num_envs: int = 8) -> None:
        self._lock = threading.Lock()
        self._num_envs = num_envs
        self._frame: np.ndarray | None = None
        self._raw_frame: np.ndarray | None = None
        self._scrolls: list[int] = [0] * num_envs
        self._active_env: int = 0
        self.env0_episode: int = 0
        self.env0_scroll: int = 0
        self.env0_features: list[float] = []
        self.env0_run_log: dict = {}
        self.agent_view: np.ndarray | None = None
        self.buffer_size: int = 0
        self.buffer_capacity: int = 0
        self.practice_rewards: list[float] = []
        self.action_counts: list[int] = [0] * 16
        self.tracker = None
        self._env_frame_buffers: list[list] = [[] for _ in range(num_envs)]  # raw frames for replay
        self._best_reward: float = float("-inf")

    def write_env(self, env_id: int, frame: np.ndarray, scroll: int = 0, raw_frame=None, features=None) -> None:
        """Write frame for a specific env. Only copies frame data for active env."""
        self._scrolls[env_id] = scroll
        # Record RAW frames for replay (without overlay)
        raw = raw_frame if raw_frame is not None else frame
        self._env_frame_buffers[env_id].append(raw.copy())
        if env_id == self._active_env:
            with self._lock:
                self._frame = frame.copy()
                self._raw_frame = raw_frame.copy() if raw_frame is not None else frame.copy()
                self.env0_scroll = scroll
                if features is not None:
                    self.env0_features = features.tolist()

    def set_active(self, env_id: int) -> None:
        self._active_env = env_id
        self.env0_scroll = self._scrolls[env_id]

    def read(self) -> np.ndarray | None:
        """Read latest frame from active env."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def on_episode_done(self, env_id: int, reward: float) -> None:
        """Called when an env finishes. Saves video if new best."""
        frames = self._env_frame_buffers[env_id]
        if reward > self._best_reward and len(frames) > 10:
            self._best_reward = reward
            self._save_best_video(frames)
        self._env_frame_buffers[env_id] = []

    def _save_best_video(self, frames: list) -> None:
        """Encode frames to MP4 in background."""
        import subprocess, cv2, threading
        from pathlib import Path
        video_path = Path("contra/web/static/best_run.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)

        def _encode():
            proc = subprocess.Popen([
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", "256x240", "-r", "30",
                "-i", "pipe:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "15",
                "-vf", "scale=768:720:flags=neighbor",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for f in frames:
                proc.stdin.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR).tobytes())
            proc.stdin.close()
            proc.wait()

        threading.Thread(target=_encode, daemon=True).start()

    # Legacy compat
    def write(self, frame: np.ndarray, scroll: int = 0) -> None:
        self.write_env(self._active_env, frame, scroll)
