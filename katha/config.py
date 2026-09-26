"""Configuration: config.yaml → validated pydantic settings, and .env → API keys.

Keys are read from the environment only when needed and are never logged.
Use `mask()` whenever a key has to be shown.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

log = logging.getLogger(__name__)


class Defaults(BaseModel):
    culture: str = "Bengali, West Bengal"
    lang: Literal["en", "hi", "bn"] = "en"


class LLMConfig(BaseModel):
    provider: Literal["gemini", "claude"] = "gemini"
    gemini_model: str = "gemini-3.8-flash"
    claude_model: str = "claude-opus-5-5"
    temperature: float = 0.7
    max_chunk_words: int = Field(2500, gt=100)


class TTSConfig(BaseModel):
    model: str = "gemini-3.8-flash-tts"
    voice_mode: Literal["designed", "prebuilt"] = "designed"
    narrator_prebuilt_voice: str = "Charon"
    sample_rate: int = 24000


class RateLimitConfig(BaseModel):
    requests_per_minute: float = Field(10, gt=0)
    max_retries: int = Field(6, ge=0)
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0


class StitchConfig(BaseModel):
    pause_between_lines_ms: int = 300
    pause_between_scenes_ms: int = 1000
    target_dbfs: float = -16.0
    mp3_bitrate: str = "192k"


class ModelPrice(BaseModel):
    input: float   # USD per 1M input tokens
    output: float  # USD per 1M output tokens


class PricingConfig(BaseModel):
    audio_tokens_per_second: float = 25
    models: dict[str, ModelPrice] = {}
    free_tier: bool = True


class PathsConfig(BaseModel):
    runs_dir: Path = Path("runs")


class Config(BaseModel):
    defaults: Defaults = Defaults()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    stitch: StitchConfig = StitchConfig()
    pricing: PricingConfig = PricingConfig()
    paths: PathsConfig = PathsConfig()

    def price_for(self, model: str) -> ModelPrice:
        try:
            return self.pricing.models[model]
        except KeyError:
            raise ConfigError(
                f"No price for model '{model}' in config.yaml → pricing.models"
            ) from None


class ConfigError(RuntimeError):
    pass


def load_config(path: Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config.model_validate(raw)
    if not cfg.paths.runs_dir.is_absolute():
        cfg.paths.runs_dir = path.parent / cfg.paths.runs_dir
    return cfg


def load_env(env_path: Path | None = None) -> None:
    """Load .env into os.environ, without overriding variables that are already set."""
    load_dotenv(env_path or PROJECT_ROOT / ".env", override=False)


KEY_HELP = {
    "GEMINI_API_KEY": "Get one at https://aistudio.google.com/apikey",
    "ANTHROPIC_API_KEY": "Needed only for --llm claude. Get one at https://console.anthropic.com",
}


def require_key(name: str) -> str:
    """Return the API key or raise a clear error. The error never contains the key."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Add it to .env (see .env.example). {KEY_HELP.get(name, '')}"
        )
    return value


def mask(secret: str | None) -> str:
    if not secret:
        return "(not set)"
    return f"set ({len(secret)} chars)"
