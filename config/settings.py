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
    death_penalty: float = -100.0
    progress_scale: float = 1.0

    # Feature flags
    hybrid_observation: bool = True    # RAM features alongside pixels
    prioritised_replay: bool = True    # PER: surprise-weighted sampling
    overlay_sprites: bool = True       # Draw enemy/bullet markers on frames


settings = Settings()
