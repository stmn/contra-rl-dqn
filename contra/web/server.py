"""FastAPI web dashboard: WebSocket for live game frames + stats + controls."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from contra.stats.tracker import StatsTracker
from contra.training.callbacks import SharedFrameBuffer

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Contra RL Dashboard")

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

app.add_middleware(NoCacheMiddleware)

_tracker: StatsTracker | None = None
_frame_buffer: SharedFrameBuffer | None = None
_controls: "TrainingControls | None" = None


class TrainingControls:
    """Thread-safe training controls accessible from web UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.paused = False
        self.restart_requested = False
        self.save_requested = False
        self.save_state_requested = False
        self.clear_state_requested = False
        self.practice_mode = False
        self.episode_length = 18_000  # 10 min safety net
        self.num_envs = 8

    def request_restart(self) -> None:
        with self._lock:
            self.restart_requested = True

    def consume_restart(self) -> bool:
        with self._lock:
            if self.restart_requested:
                self.restart_requested = False
                return True
            return False

    def toggle_pause(self) -> bool:
        with self._lock:
            self.paused = not self.paused
            return self.paused

    def request_save(self) -> None:
        with self._lock:
            self.save_requested = True

    def consume_save(self) -> bool:
        with self._lock:
            if self.save_requested:
                self.save_requested = False
                return True
            return False

    def request_save_state(self) -> None:
        with self._lock:
            self.save_state_requested = True

    def consume_save_state(self) -> bool:
        with self._lock:
            if self.save_state_requested:
                self.save_state_requested = False
                return True
            return False

    def request_clear_state(self) -> None:
        with self._lock:
            self.clear_state_requested = True
            self.practice_mode = False

    def consume_clear_state(self) -> bool:
        with self._lock:
            if self.clear_state_requested:
                self.clear_state_requested = False
                return True
            return False


def init(tracker: StatsTracker, frame_buffer: SharedFrameBuffer) -> None:
    global _tracker, _frame_buffer, _controls
    _tracker = tracker
    _frame_buffer = frame_buffer
    _controls = TrainingControls()


def get_controls() -> TrainingControls | None:
    return _controls


_LOCAL_PREFIXES = ("127.", "192.168.", "10.", "172.16.", "::1", "localhost")

def _is_local(request) -> bool:
    client = request.client.host if request.client else ""
    forwarded = request.headers.get("cf-connecting-ip", request.headers.get("x-forwarded-for", ""))
    # If behind Cloudflare, cf-connecting-ip is the real IP
    ip = forwarded.split(",")[0].strip() if forwarded else client
    return any(ip.startswith(p) for p in _LOCAL_PREFIXES) or not forwarded


@app.get("/api/is-admin")
async def is_admin(request: Request):
    return {"admin": _is_local(request)}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/stats")
async def get_stats():
    if _tracker is None:
        return {"error": "not initialized"}
    snap = _tracker.snapshot()
    d = _snap_to_dict(snap)
    if _controls:
        d["paused"] = _controls.paused
    return d


@app.get("/api/level")
async def get_level_data():
    if not _tracker:
        return {"death_heatmap": [], "max_scroll": 0, "env0_scroll": 0, "time_since_pb": 0}
    return {
        "death_positions": _tracker.death_positions(),
        "max_scroll": max(_tracker.max_scroll(), _frame_buffer.env0_scroll if _frame_buffer else 0),
        "env0_scroll": _frame_buffer.env0_scroll if _frame_buffer else 0,
        "time_since_pb": round(_tracker.time_since_pb()),
    }


@app.get("/api/history")
async def get_history():
    if _tracker is None:
        return {"reward_history": [], "top_runs": []}
    return {
        "reward_history": _tracker.reward_history(5000),
        "survival_history": _tracker.survival_history(5000),
        "top_runs": _tracker.top_runs(10),
    }


@app.post("/api/restart")
async def restart_episode(request: Request):
    if not _is_local(request):
        return JSONResponse({"ok": False}, status_code=403)
    if _controls:
        _controls.request_restart()
        return {"ok": True, "message": "Restart requested"}
    return JSONResponse({"ok": False}, status_code=503)


@app.post("/api/pause")
async def toggle_pause(request: Request):
    if not _is_local(request):
        return JSONResponse({"ok": False}, status_code=403)
    if _controls:
        paused = _controls.toggle_pause()
        return {"ok": True, "paused": paused}
    return JSONResponse({"ok": False}, status_code=503)


@app.post("/api/save")
async def save_checkpoint(request: Request):
    if not _is_local(request):
        return JSONResponse({"ok": False}, status_code=403)
    if _controls:
        _controls.request_save()
        return {"ok": True, "message": "Save requested"}
    return JSONResponse({"ok": False}, status_code=503)


@app.post("/api/save-state")
async def save_game_state(request: Request):
    if not _is_local(request):
        return JSONResponse({"ok": False}, status_code=403)
    if _controls:
        _controls.request_save_state()
        return {"ok": True, "message": "Game state saved — practice mode ON"}
    return JSONResponse({"ok": False}, status_code=503)


@app.post("/api/clear-state")
async def clear_game_state(request: Request):
    if not _is_local(request):
        return JSONResponse({"ok": False}, status_code=403)
    if _controls:
        _controls.request_clear_state()
        return {"ok": True, "message": "Game state cleared — practice mode OFF"}
    return JSONResponse({"ok": False}, status_code=503)


@app.get("/api/config")
async def get_config():
    conf = {
        "features": {
            "hybrid_observation": settings.hybrid_observation,
            "prioritised_replay": settings.prioritised_replay,
            "overlay_sprites": settings.overlay_sprites,
        },
        "rewards": {
            "death_penalty": settings.death_penalty,
            "progress_scale": settings.progress_scale,
        },
        "training": {
            "device": settings.device,
            "total_timesteps": settings.total_timesteps,
        },
    }
    if _frame_buffer and hasattr(_frame_buffer, 'trainer_config'):
        conf["dqn"] = _frame_buffer.trainer_config
    return conf


@app.get("/api/best-replay")
async def get_best_replay():
    import json
    path = Path("logs/best_replay.json")
    if not path.exists():
        return {"reward": 0, "actions": []}
    return json.loads(path.read_text())


@app.post("/api/play-best")
async def play_best_run():
    """Return URL to pre-recorded best run video."""
    video_path = STATIC_DIR / "best_run.mp4"
    if not video_path.exists():
        return JSONResponse({"ok": False, "message": "No best run recorded yet"}, status_code=404)
    # Add timestamp to bust cache
    import os
    mtime = int(os.path.getmtime(video_path))
    return {"ok": True, "url": f"/static/best_run.mp4?t={mtime}"}


@app.post("/api/settings")
async def update_settings(request: Request, data: dict):
    if not _is_local(request):
        return JSONResponse({"ok": False}, status_code=403)
    if not _controls:
        return JSONResponse({"ok": False}, status_code=503)
    if "episode_length" in data:
        val = int(data["episode_length"])
        if 100 <= val <= 50_000:
            _controls.episode_length = val
            if _tracker:
                _tracker._episode_length = val
    return {"ok": True, "episode_length": _controls.episode_length}


@app.get("/api/settings")
async def get_settings():
    if not _controls:
        return {"episode_length": 600}
    return {"episode_length": _controls.episode_length}


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    """High-frequency stream: JPEG frames with scroll/episode header."""
    import struct
    await ws.accept()
    last_id = None
    try:
        while True:
            if _frame_buffer:
                frame = _frame_buffer.read()
                if frame is not None:
                    fid = frame.ctypes.data
                    if fid != last_id:
                        last_id = fid
                        # Main preview: raw frame (color)
                        with _frame_buffer._lock:
                            raw = _frame_buffer._raw_frame
                        if raw is not None:
                            main_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                        else:
                            main_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        _, main_buf = cv2.imencode(".jpg", main_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])

                        # Agent view: overlay frame resized to 128x128 grayscale
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
                        _, agent_buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])

                        # Header: scroll(u16) + episode(u16) + main_size(u32) + agent_size(u32)
                        header = struct.pack("<HHII",
                            min(_frame_buffer.env0_scroll, 65535),
                            min(_frame_buffer.env0_episode, 65535),
                            len(main_buf),
                            len(agent_buf))
                        await ws.send_bytes(header + main_buf.tobytes() + agent_buf.tobytes())
            await asyncio.sleep(1 / 60)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/stats")
async def ws_stats(ws: WebSocket):
    """Low-frequency stream: stats at 2fps."""
    await ws.accept()
    try:
        while True:
            msg = {}
            if _tracker:
                d = _snap_to_dict(_tracker.snapshot())
                if _controls:
                    d["paused"] = _controls.paused
                    d["practice"] = _controls.practice_mode
                if _frame_buffer:
                    d["env0_episode"] = _frame_buffer.env0_episode
                    d["features"] = _frame_buffer.env0_features
                    d["action_counts"] = _frame_buffer.action_counts
                msg["stats"] = d
            await ws.send_json(msg)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


def _snap_to_dict(snap) -> dict:
    return {
        "episode": snap.episode,
        "current_reward": round(snap.current_reward, 2),
        "best_reward": round(snap.best_reward, 2),
        "best_level": snap.best_level,
        "deaths_this_run": snap.deaths_this_run,
        "total_deaths": snap.total_deaths,
        "training_time": time.time() - snap.training_start,
        "timesteps": snap.timesteps,
        "total_timesteps": snap.total_timesteps,
        "fps": round(snap.fps, 1),
        "generation": snap.generation,
        "reward_scroll": snap.last_reward_scroll,
        "reward_death": snap.last_reward_death,
        "death_count": snap.last_death_count,
        "timeout_pct": snap.timeout_pct,
        "rollback_count": snap.rollback_count,
        "last_rollback_ago": snap.last_rollback_ago,
        "last_autosave_ago": snap.last_autosave_ago,
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
