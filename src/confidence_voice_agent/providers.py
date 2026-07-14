from livekit.agents import AgentSession
from livekit.plugins import deepgram, elevenlabs, openai, silero

from confidence_voice_agent.config import Settings


class ProviderConfigurationError(RuntimeError):
    pass


def create_agent_session(settings: Settings) -> AgentSession:
    return AgentSession(
        stt=create_stt(settings),
        llm=create_llm(settings),
        tts=create_tts(settings),
        vad=silero.VAD.load(),
    )


def create_stt(settings: Settings) -> deepgram.STT:
    return deepgram.STT(
        model=settings.deepgram_model,
        language="en",
        interim_results=True,
        api_key=settings.secret_value(settings.deepgram_api_key),
    )


def create_llm(settings: Settings) -> openai.LLM:
    return openai.LLM(
        model=settings.openai_model,
        api_key=settings.secret_value(settings.openai_api_key),
        temperature=0.2,
    )


def create_tts(settings: Settings) -> elevenlabs.TTS:
    voice_id = settings.elevenlabs_voice_id.strip()
    if not voice_id:
        raise ProviderConfigurationError("ELEVENLABS_VOICE_ID must not be empty")

    return elevenlabs.TTS(
        voice_id=voice_id,
        model=settings.elevenlabs_model,
        api_key=settings.secret_value(settings.elevenlabs_api_key),
    )
