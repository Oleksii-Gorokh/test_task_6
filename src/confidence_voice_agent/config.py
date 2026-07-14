import logging
from functools import lru_cache
from typing import Self
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    livekit_url: str = Field(alias="LIVEKIT_URL")
    livekit_api_key: SecretStr = Field(alias="LIVEKIT_API_KEY")
    livekit_api_secret: SecretStr = Field(alias="LIVEKIT_API_SECRET")
    deepgram_api_key: SecretStr = Field(alias="DEEPGRAM_API_KEY")
    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    elevenlabs_api_key: SecretStr = Field(alias="ELEVENLABS_API_KEY")
    deepgram_model: str = Field(default="nova-3", alias="DEEPGRAM_MODEL")
    deepgram_language: str = Field(default="en-US", alias="DEEPGRAM_LANGUAGE")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    elevenlabs_model: str = Field(default="eleven_flash_v2_5", alias="ELEVENLABS_MODEL")
    elevenlabs_voice_id: str = Field(default="hpp4J3VqNfWAUOO0d1Us", alias="ELEVENLABS_VOICE_ID")
    low_confidence_threshold: float = Field(default=0.6, alias="LOW_CONFIDENCE_THRESHOLD")
    high_confidence_threshold: float = Field(default=0.8, alias="HIGH_CONFIDENCE_THRESHOLD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError("LOG_LEVEL must be a standard Python logging level")
        return normalized

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        for name, secret in {
            "LIVEKIT_API_KEY": self.livekit_api_key,
            "LIVEKIT_API_SECRET": self.livekit_api_secret,
            "DEEPGRAM_API_KEY": self.deepgram_api_key,
            "OPENAI_API_KEY": self.openai_api_key,
            "ELEVENLABS_API_KEY": self.elevenlabs_api_key,
        }.items():
            if not secret.get_secret_value().strip():
                raise ValueError(f"{name} must not be empty")
        parsed_url = urlparse(self.livekit_url)
        if parsed_url.scheme not in {"ws", "wss"} or not parsed_url.netloc:
            raise ValueError("LIVEKIT_URL must be a valid ws:// or wss:// URL")
        for name, option in {
            "DEEPGRAM_MODEL": self.deepgram_model,
            "DEEPGRAM_LANGUAGE": self.deepgram_language,
            "OPENAI_MODEL": self.openai_model,
            "ELEVENLABS_MODEL": self.elevenlabs_model,
            "ELEVENLABS_VOICE_ID": self.elevenlabs_voice_id,
        }.items():
            if not option.strip():
                raise ValueError(f"{name} must not be empty")
        if not 0 <= self.low_confidence_threshold <= 1:
            raise ValueError("LOW_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if not 0 <= self.high_confidence_threshold <= 1:
            raise ValueError("HIGH_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if self.low_confidence_threshold >= self.high_confidence_threshold:
            raise ValueError(
                "LOW_CONFIDENCE_THRESHOLD must be lower than HIGH_CONFIDENCE_THRESHOLD"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings.model_validate({})
