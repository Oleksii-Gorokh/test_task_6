from functools import lru_cache
from typing import Any, Self

from pydantic import Field, SecretStr, model_validator
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
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    elevenlabs_model: str = Field(default="eleven_flash_v2_5", alias="ELEVENLABS_MODEL")
    elevenlabs_voice_id: str = Field(default="hpp4J3VqNfWAUOO0d1Us", alias="ELEVENLABS_VOICE_ID")
    low_confidence_threshold: float = Field(default=0.6, alias="LOW_CONFIDENCE_THRESHOLD")
    high_confidence_threshold: float = Field(default=0.8, alias="HIGH_CONFIDENCE_THRESHOLD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if not 0 <= self.low_confidence_threshold <= 1:
            raise ValueError("LOW_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if not 0 <= self.high_confidence_threshold <= 1:
            raise ValueError("HIGH_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if self.low_confidence_threshold >= self.high_confidence_threshold:
            raise ValueError(
                "LOW_CONFIDENCE_THRESHOLD must be lower than HIGH_CONFIDENCE_THRESHOLD"
            )
        return self

    def secret_value(self, value: SecretStr) -> str:
        return value.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    settings_kwargs: dict[str, Any] = {}
    return Settings(**settings_kwargs)
