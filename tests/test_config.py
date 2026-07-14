import pytest
from pydantic import ValidationError

from confidence_voice_agent.config import Settings


def valid_settings(**overrides: object) -> Settings:
    from typing import Any

    data: dict[str, Any] = {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "DEEPGRAM_API_KEY": "deepgram-key",
        "OPENAI_API_KEY": "openai-key",
        "ELEVENLABS_API_KEY": "elevenlabs-key",
    }
    data.update(overrides)
    return Settings.model_validate(data)


def test_settings_load_valid_minimal_config() -> None:
    settings = valid_settings()

    assert settings.livekit_url == "wss://example.livekit.cloud"
    assert settings.deepgram_model == "nova-3"
    assert settings.deepgram_language == "en-US"
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.low_confidence_threshold == 0.6
    assert settings.high_confidence_threshold == 0.8


def test_settings_require_provider_keys() -> None:
    with pytest.raises(ValidationError) as error:
        valid_settings(DEEPGRAM_API_KEY=None)

    assert "DEEPGRAM_API_KEY" in str(error.value)


@pytest.mark.parametrize(
    "field",
    [
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
    ],
)
def test_settings_reject_empty_secret_values(field: str) -> None:
    with pytest.raises(ValidationError) as error:
        valid_settings(**{field: " "})

    assert f"{field} must not be empty" in str(error.value)


@pytest.mark.parametrize("url", [" ", "https://example.com", "ws://"])
def test_settings_reject_invalid_livekit_url(url: str) -> None:
    with pytest.raises(ValidationError) as error:
        valid_settings(LIVEKIT_URL=url)

    assert "LIVEKIT_URL must be a valid ws:// or wss:// URL" in str(error.value)


@pytest.mark.parametrize(
    "field",
    [
        "DEEPGRAM_MODEL",
        "DEEPGRAM_LANGUAGE",
        "OPENAI_MODEL",
        "ELEVENLABS_MODEL",
        "ELEVENLABS_VOICE_ID",
    ],
)
def test_settings_reject_empty_provider_options(field: str) -> None:
    with pytest.raises(ValidationError) as error:
        valid_settings(**{field: " "})

    assert f"{field} must not be empty" in str(error.value)


def test_settings_normalizes_valid_log_level() -> None:
    assert valid_settings(LOG_LEVEL="debug").log_level == "DEBUG"


def test_settings_rejects_invalid_log_level() -> None:
    with pytest.raises(ValidationError) as error:
        valid_settings(LOG_LEVEL="verbose")

    assert "LOG_LEVEL must be a standard Python logging level" in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("LOW_CONFIDENCE_THRESHOLD", -0.1),
        ("LOW_CONFIDENCE_THRESHOLD", 1.1),
        ("HIGH_CONFIDENCE_THRESHOLD", -0.1),
        ("HIGH_CONFIDENCE_THRESHOLD", 1.1),
    ],
)
def test_settings_reject_thresholds_outside_unit_interval(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        valid_settings(**{field: value})


def test_settings_reject_low_threshold_greater_than_or_equal_to_high_threshold() -> None:
    with pytest.raises(ValidationError) as error:
        valid_settings(LOW_CONFIDENCE_THRESHOLD=0.8, HIGH_CONFIDENCE_THRESHOLD=0.8)

    assert "LOW_CONFIDENCE_THRESHOLD must be lower" in str(error.value)
