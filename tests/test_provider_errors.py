import pytest

from confidence_voice_agent.providers import ProviderConfigurationError, create_tts
from tests.test_config import valid_settings


def test_tts_provider_rejects_empty_voice_id() -> None:
    settings = valid_settings(ELEVENLABS_VOICE_ID=" ")

    with pytest.raises(ProviderConfigurationError) as error:
        create_tts(settings)

    assert "ELEVENLABS_VOICE_ID" in str(error.value)
