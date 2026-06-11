from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", env_prefix="KOTOMKA_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: Path = Path("data")
    llm_provider: str = "auto"
    stt_provider: str = "fake"
    frame_interval_seconds: int = 15
    frame_max_gap_seconds: int = 60
    frame_plateau_min_dwell_seconds: float = 3.0
    frame_plateau_hash_distance: int = 3
    # 0.0 disables the blur gate; calibrate on real talks before enabling.
    frame_blur_threshold: float = 0.0
    max_video_duration_seconds: int = 2 * 60 * 60
    max_frames_for_llm: int = 24
    max_selected_frames: int = 24
    selected_frame_min_gap_seconds: int = 20
    openai_model: str = "gpt-4.1"
    openai_scoring_model: str | None = None
    codex_model: str = "gpt-5.4"
    codex_scoring_model: str | None = None
    scoring_image_detail: str = "low"
    report_image_detail: str = "high"
    report_max_images: int = 16
    report_single_pass_max_chars: int = 24000
    report_chunk_target_seconds: int = 600
    transcript_excerpt_margin_seconds: int = 30
    transcript_low_confidence_threshold: float = 0.5
    assessment_enabled: bool = True
    assessment_web_search: bool = False
    codex_auth_file: Path = Field(
        default=Path("data/codex_subscription_auth.json"),
        validation_alias=AliasChoices("KOTOMKA_CODEX_AUTH_FILE", "CODEX_SUBSCRIPTION_AUTH_FILE"),
    )
    assemblyai_poll_seconds: float = 3.0
    citation_snap_tolerance_seconds: float = 5.0
    stt_keyterms_max: int = 200

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    return settings
