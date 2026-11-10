"""Application configuration & preset profiles for subllm."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SubtitleProfile(StrEnum):
    """Preset profiles tailored for different video genres."""

    CINEMA = "cinema"  # Movie / TV drama: strict CPS/CPL, character voice, scene-gap aware
    ASR_YOUTUBE = "asr"  # YouTube / Whisper ASR: aggressive cross-cue sentence merging
    LECTURE = "lecture"  # Tech talk / documentary: terminology strictness, wider CPL


class ProfileConfig(BaseSettings):
    """Timing and segmentation thresholds for a specific subtitle profile."""

    name: SubtitleProfile = SubtitleProfile.CINEMA
    max_cps: float = Field(default=4.5, description="Max characters per second (Chinese target)")
    max_cpl: int = Field(default=18, description="Max characters per single line")
    scene_gap_sec: float = Field(
        default=3.0, description="Silence gap (seconds) that indicates a scene/topic boundary"
    )
    merge_gap_sec: float = Field(
        default=0.8,
        description="Max gap (seconds) to link split sentences across adjacent cues",
    )
    lines_per_block: int = Field(default=15, description="Target cues per translation block")
    enable_bible_pass: bool = Field(
        default=True,
        description="Run Pass 1 to extract MovieBible (synopsis, characters, glossary)",
    )
    enable_condense_pass: bool = Field(
        default=True, description="Run Pass 3 to auto-condense cues exceeding max_cps or max_cpl"
    )

    @classmethod
    def from_preset(cls, preset: SubtitleProfile) -> "ProfileConfig":
        """Factory method returning tuned parameters for each preset profile."""
        if preset == SubtitleProfile.CINEMA:
            return cls(
                name=SubtitleProfile.CINEMA,
                max_cps=4.5,
                max_cpl=18,
                scene_gap_sec=3.0,
                merge_gap_sec=0.8,
                lines_per_block=15,
            )
        if preset == SubtitleProfile.ASR_YOUTUBE:
            return cls(
                name=SubtitleProfile.ASR_YOUTUBE,
                max_cps=5.5,
                max_cpl=22,
                scene_gap_sec=2.5,
                merge_gap_sec=1.5,
                lines_per_block=20,
            )
        return cls(
            name=SubtitleProfile.LECTURE,
            max_cps=5.5,
            max_cpl=24,
            scene_gap_sec=4.0,
            merge_gap_sec=1.0,
            lines_per_block=18,
        )


class Settings(BaseSettings):
    """Global settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    subllm_provider: Literal[
        "gemini", "openai", "deepseek", "anthropic", "openrouter", "mistral", "groq", "ollama"
    ] = "gemini"
    subllm_model: str = "gemini-2.5-flash"
    subllm_api_endpoint: str = ""

    gemini_api_key: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    mistral_api_key: str = ""
    groq_api_key: str = ""

    subllm_max_cps: float = 4.5
    subllm_max_cpl: int = 18
    subllm_scene_gap_sec: float = 3.0
    subllm_merge_gap_sec: float = 0.8
    subllm_lines_per_block: int = 15

    cache_db_path: Path = Path(".subllm_cache/checkpoints.db")

    @property
    def active_api_keys(self) -> list[str]:
        """Return comma-separated API keys for the active provider as a clean list."""
        key_map = {
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
            "deepseek": self.deepseek_api_key,
            "anthropic": self.anthropic_api_key,
            "openrouter": self.openrouter_api_key,
            "mistral": self.mistral_api_key,
            "groq": self.groq_api_key,
            "ollama": "ollama-local",
        }
        raw = key_map.get(self.subllm_provider, "")
        return [k.strip() for k in raw.split(",") if k.strip()]


# Curated model catalog per provider for UI dropdowns and CLI help
PROVIDER_MODELS: dict[str, list[str]] = {
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "anthropic": [
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    "openai": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
    ],
    "openrouter": [
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-chat-v3-0324",
        "deepseek/deepseek-r1",
        "google/gemini-2.5-pro-preview",
        "google/gemini-2.5-flash-preview",
        "openai/gpt-4.1",
        "qwen/qwen-2.5-72b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-large-2411",
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-small-latest",
        "ministral-8b-latest",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "qwen-qwq-32b",
    ],
    "ollama": [
        "qwen2.5:14b-instruct",
        "qwen2.5:32b",
        "gemma3:27b",
        "llama3.3:70b",
        "deepseek-r1:14b",
        "mistral-small:24b",
    ],
}

settings = Settings()

