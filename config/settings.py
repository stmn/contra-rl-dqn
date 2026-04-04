from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    rom_path: Path = Path("./roms/contra.nes")
    integration_path: Path = Path("./integrations")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    # Training
    device: str = "cpu"
    total_timesteps: int = 100_000_000

    # Web
    web_port: int = 41918

    # Rewards
    death_penalty: float = -500.0
    progress_scale: float = 1.0

    # Feature flags
    hybrid_observation: bool = True    # RAM features alongside pixels
    prioritised_replay: bool = True    # PER: surprise-weighted sampling
    overlay_sprites: bool = True       # Draw enemy/bullet markers on frames
    dueling_dqn: bool = True           # Separate V(state) + A(action) streams
    noisy_nets: bool = True            # Noisy linear layers instead of epsilon-greedy
    n_step_returns: int = 3            # N-step returns (1 = standard, 3 = recommended)
    huber_loss: bool = True            # Huber loss instead of MSE (robust to outliers)
    gradient_clip: float = 10.0        # Max gradient norm (0 = disabled)


settings = Settings()
