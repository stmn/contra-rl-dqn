"""Main entry point: DQN training + web dashboard."""

import logging
import signal
import sys
import threading
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from contra.env.contra_env import ContraEnv
from contra.env.wrappers import wrap_contra
from contra.stats.tracker import StatsTracker
from contra.training.callbacks import SharedFrameBuffer
from contra.training.dqn import DQNTrainer
from contra.web.server import app, init as web_init, get_controls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("contra")


def find_latest_checkpoint() -> Path | None:
    d = settings.checkpoint_dir
    if not d.exists():
        return None
    manual = d / "contra_dqn_manual.pt"
    if manual.exists():
        return manual
    auto = d / "auto_best.pt"
    if auto.exists():
        return auto
    return None


def main() -> None:
    log.info("=== CONTRA RL (DQN) — Starting ===")

    # Shared state
    tracker = StatsTracker(
        save_path=settings.log_dir / "stats.json",
        total_timesteps=settings.total_timesteps,
    )
    frame_buffer = SharedFrameBuffer(num_envs=1)
    frame_buffer.tracker = tracker
    frame_buffer._best_reward = tracker._best_reward  # don't overwrite existing best video

    # Web dashboard
    web_init(tracker, frame_buffer)
    get_controls().episode_length = tracker._episode_length
    web_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "0.0.0.0", "port": settings.web_port, "log_level": "warning"},
        daemon=True,
        name="web",
    )
    web_thread.start()
    log.info(f"Web dashboard: http://localhost:{settings.web_port}")

    # Single env with frame capture
    env = ContraEnv(rom_path=settings.rom_path, overlay_sprites=True)
    env = wrap_contra(env, frame_buffer=frame_buffer, env_id=0)

    # DQN Trainer
    def on_step(global_step, fps):
        tracker.update_step(global_step, fps)

    def on_episode(reward, info):
        tracker.on_episode_end(reward, info.get("level", 0), info.get("deaths", 0), info)

    def on_save():
        settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = str(settings.checkpoint_dir / "contra_dqn_manual.pt")
        trainer.save(path)
        tracker.save()
        log.info(f"Manual checkpoint saved: {path}")

    trainer = DQNTrainer(
        env,
        on_step=on_step,
        on_episode=on_episode,
        controls=get_controls(),
        on_save=on_save,
        frame_buffer=frame_buffer,
    )

    # Expose trainer config to web API (cast numpy types to native Python)
    frame_buffer.trainer_config = {
        "lr": float(trainer.lr),
        "gamma": float(trainer.gamma),
        "batch_size": int(trainer.batch_size),
        "buffer_size": int(trainer.buffer_size),
        "train_freq": int(trainer.train_freq),
        "target_update_freq": int(trainer.target_update_freq),
        "epsilon_start": float(trainer.epsilon_start),
        "epsilon_end": float(trainer.epsilon_end),
        "epsilon_decay": int(trainer.epsilon_decay),
        "n_actions": int(trainer.n_actions),
        "hybrid": bool(trainer._hybrid),
        "per": bool(trainer._per),
        "frame_skip": int(env.unwrapped._frame_skip),
        "max_episode_steps": int(env.unwrapped._max_steps),
    }

    checkpoint = find_latest_checkpoint()
    if checkpoint:
        log.info(f"Resuming from: {checkpoint}")
        trainer.load(str(checkpoint))
        # Restore peak avg from saved history so auto-save doesn't spam
        history = tracker.reward_history(200)
        if len(history) >= 30:
            import numpy as _np
            trainer._peak_avg = max(
                _np.mean(history[i:i+30]) for i in range(len(history) - 29)
            )
            log.info(f"Restored peak avg: {trainer._peak_avg:.0f}")
    else:
        log.info("Starting fresh training")

    # Graceful shutdown
    def shutdown(signum, frame):
        log.info("Shutdown signal received...")
        tracker.save()
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        trainer.train()
    finally:
        env.close()
        log.info("=== CONTRA RL (DQN) — Stopped ===")


if __name__ == "__main__":
    main()
